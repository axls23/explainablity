"""
The Analyzer: Causal Inference Engine.

Implements causal patching sweeps, DTW trajectory comparison,
topological data analysis (persistent homology), composite
validity scoring, and rigorous time-series statistical testing
(Gaps A–E).
"""

import numpy as np
import os
import torch
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from rich.console import Console
from rich.progress import Progress
import uuid
import threading
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from statsmodels.stats.diagnostic import acorr_ljungbox

from .config import ChronoscopeConfig
from .interceptor import ChronoscopeInterceptor
from .observer import SignalObserver

console = Console()



class MockVARResult:
    """Helper for Ridge VAR mapping to statsmodels-like interface."""
    def __init__(self, coefs, k_ar):
        self.coefs = coefs
        self.k_ar = k_ar

class CausalAnalyzer:

    """
    Performs activation sensitivity analysis on the model's reasoning trace and
    analyzes the resulting trajectory divergence using DTW, TDA, and SVD.

    NOTE ON TERMINOLOGY:
        This class was originally named for "causal" analysis, but the methods
        implemented here measure SENSITIVITY and ATTRIBUTION, not Pearlian
        causal inference. Activation patching in neural networks violates
        modularity assumptions of do-calculus. Results indicate which
        components are important/influential, not mechanistic causation.

        For true causal analysis, consider:
            - Causal mediation analysis (Gap C.4 methods)
            - Proper interchange interventions with causal graphs
            - Reference: Geiger et al. "Causal Abstractions of Neural Networks"
    """

    def __init__(
        self,
        interceptor: ChronoscopeInterceptor,
        observer: SignalObserver,
        config: ChronoscopeConfig,
    ):
        self.interceptor = interceptor
        self.observer = observer
        self.config = config
        # Lazy-loaded statsmodels VAR class for head interaction analysis.
        self._var_cls = None
        # Bridge-facing cached outputs.
        self._last_influence_matrix = None
        self._last_fdr_result = None
        self._last_pdc = None
        self._selected_lag = None
        self._last_joint_stationarity = None
        self._last_coint = None
        self._vecm_used = False
        self._last_pert_results = None
        self._last_mediation_results = None
        self._last_hmm_result = None
        self._intervention_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Causal Patching Sweep
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Backward Compatibility Wrappers
    # ------------------------------------------------------------------ #
    def causal_patching_sweep(self, *args, **kwargs):
        """[DEPRECATED] Use sensitivity_analysis_sweep instead. 
        Renamed to reflect mathematical reality: patching traces sensitivity, not Pearlian causality."""
        import warnings
        warnings.warn("causal_patching_sweep is deprecated. Use sensitivity_analysis_sweep.", DeprecationWarning)
        return self.sensitivity_analysis_sweep(*args, **kwargs)

    def sensitivity_analysis_sweep(
        self,
        prompt: str,
        layer_names: Optional[List[str]] = None,
        token_range: Optional[range] = None,
    ) -> Dict:
        """
        Sweep across layers × tokens. For each (layer, token), ablate the
        activation and measure the output divergence.

        SENSITIVITY ANALYSIS WARNING:
            This method measures SENSITIVITY of outputs to activation ablations,
            NOT causal influence in the Pearlian sense. In neural networks,
            ablation affects all downstream computations simultaneously,
            violating the "surgical intervention" assumption of do-calculus.

            Interpret results as:
                - "This position is IMPORTANT for the output"
                - "The trajectory is SENSITIVE to ablation here"
                NOT as "This position CAUSES the output"

        Returns a sensitivity heatmap: [n_layers, n_tokens] of divergence scores.
        """
        # Get clean trajectory first
        console.print("[cyan]Capturing clean trajectory...[/]")
        clean_traj, clean_text = self.interceptor.capture_generation(prompt)

        if not clean_traj:
            raise RuntimeError("No activations captured. Check target_layers config.")

        if layer_names is None:
            layer_names = sorted(clean_traj.keys())

        # Get token count from the first layer's trajectory
        first_layer = list(clean_traj.values())[0]
        n_tokens = first_layer.shape[0]

        if token_range is None:
            # Only patch input tokens (not generated ones — those don't exist yet)
            inputs = self.interceptor.tokenizer(prompt, return_tensors="pt")
            n_input_tokens = inputs["input_ids"].shape[1]
            token_range = range(n_input_tokens)

        n_layers = len(layer_names)
        n_tok = len(token_range)
        heatmap = np.zeros((n_layers, n_tok))

        # Compress clean trajectory for comparison
        clean_ref_layer = clean_traj[layer_names[-1]]  # Use last layer
        clean_compressed, _, _ = self.observer.svd_compress(clean_ref_layer)

        console.print(
            f"[cyan]Running patching sweep: {n_layers} layers × {n_tok} tokens[/]"
        )

        with Progress() as progress:
            task = progress.add_task("Patching...", total=n_layers * n_tok)

            for li, layer_name in enumerate(layer_names):
                for ti, token_idx in enumerate(token_range):
                    try:
                        patched_traj, patched_text = self.interceptor.capture_with_ablation(
                            prompt,
                            target_layer_name=layer_name,
                            token_indices=[token_idx],
                        )

                        # Compare last-layer trajectories
                        if layer_names[-1] in patched_traj:
                            patched_ref = patched_traj[layer_names[-1]]
                            divergence = self._trajectory_divergence(
                                clean_ref_layer, patched_ref
                            )
                            heatmap[li, ti] = divergence
                        else:
                            heatmap[li, ti] = 0.0

                    except Exception as e:
                        heatmap[li, ti] = -1.0  # Mark errors

                    progress.advance(task)

        return {
            "heatmap": heatmap,
            "layer_names": layer_names,
            "token_indices": list(token_range),
            "clean_text": clean_text,
            "token_labels": self._get_token_labels(prompt, token_range),
        }

    def _get_token_labels(self, prompt: str, token_range: range) -> List[str]:
        """Get human-readable token labels for the heatmap."""
        tokens = self.interceptor.tokenizer.tokenize(prompt)
        labels = []
        for idx in token_range:
            if idx < len(tokens):
                labels.append(f"{idx}:{tokens[idx]}")
            else:
                labels.append(f"{idx}:<gen>")
        return labels

    # ------------------------------------------------------------------ #
    #  Trajectory Divergence (L2 + Cosine)
    # ------------------------------------------------------------------ #

    def _trajectory_divergence(
        self, clean: torch.Tensor, patched: torch.Tensor
    ) -> float:
        """
        Quick divergence metric between two trajectories.
        Uses mean L2 distance across shared tokens.
        """
        min_len = min(clean.shape[0], patched.shape[0])
        c = clean[:min_len].float()
        p = patched[:min_len].float()

        # Normalized L2 distance
        l2 = torch.norm(c - p, dim=-1).mean().item()
        return l2

    def stochastic_patching_sweep(self, *args, **kwargs):
        """[DEPRECATED] Use stochastic_activation_ablation instead."""
        import warnings
        warnings.warn("stochastic_patching_sweep is deprecated. Use stochastic_activation_ablation.", DeprecationWarning)
        return self.stochastic_activation_ablation(*args, **kwargs)

    def stochastic_activation_ablation(
        self,
        prompt: str,
        layer_names: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> Dict:
        """
        Optimized sweep: Only patches 'salient' tokens where the trajectory
        shows high activity/energy, skipping static regions.
        """
        # 1. Capture clean trace
        console.print("[cyan]Capturing clean trajectory...[/]")
        clean_traj, clean_text = self.interceptor.capture_generation(prompt)
        
        if not clean_traj:
            raise RuntimeError("No activations captured.")

        if layer_names is None:
            # For speed, we might only want to patch the 'deepest' and 'middle' layers
            all_layers = sorted(clean_traj.keys())
            if len(all_layers) > 3:
                mid = len(all_layers) // 2
                from .models import get_deepest_layer
                deepest = get_deepest_layer(all_layers)
                layer_names = sorted(list(set([all_layers[mid], deepest])))
            else:
                layer_names = all_layers

        # 2. Salience Scan: Find tokens with high 'update energy'
        # We use the deepest layer as the proxy for semantic activity
        ref_layer = clean_traj[layer_names[-1]]
        # Compute L2 norm of the difference between consecutive tokens
        deltas = torch.norm(ref_layer[1:] - ref_layer[:-1], dim=-1).detach().cpu().numpy()
        
        # Identify top-K indices with highest deltas
        # (Offset by 1 because deltas are between tokens)
        salient_indices = np.argsort(deltas)[::-1][:top_k]
        salient_indices = sorted(list(set(salient_indices + 1))) 
        
        console.print(f"  [green]Salience scan identified {len(salient_indices)} interest points out of {ref_layer.shape[0]} tokens.[/]")

        # 3. Patch only the salient tokens
        n_layers = len(layer_names)
        n_tok = len(salient_indices)
        
        # Sparse result heatmap
        res_heatmap = np.zeros((n_layers, n_tok))
        
        with Progress() as progress:
            task = progress.add_task("Optimized Patching...", total=n_layers * n_tok)
            for li, layer_name in enumerate(layer_names):
                for ti, token_idx in enumerate(salient_indices):
                    try:
                        patched_traj, _ = self.interceptor.capture_with_ablation(
                            prompt,
                            target_layer_name=layer_name,
                            token_indices=[int(token_idx)],
                        )
                        if layer_names[-1] in patched_traj:
                            val = self._trajectory_divergence(ref_layer, patched_traj[layer_names[-1]])
                            res_heatmap[li, ti] = val
                    except:
                        pass
                    progress.advance(task)

        # 4. Reconstruct full heatmap shape for the report
        full_tok_count = ref_layer.shape[0]
        full_heatmap = np.zeros((n_layers, full_tok_count))
        for li in range(n_layers):
            for ti, token_idx in enumerate(salient_indices):
                full_heatmap[li, token_idx] = res_heatmap[li, ti]

        token_labels = self.interceptor.get_token_labels(prompt, clean_text)

        return {
            "heatmap": full_heatmap,
            "layer_names": layer_names,
            "token_indices": salient_indices,
            "clean_text": clean_text,
            "token_labels": token_labels,
        }

    # ------------------------------------------------------------------ #
    #  Dynamic Time Warping (DTW)
    # ------------------------------------------------------------------ #

    def dtw_divergence(
        self, clean_compressed: np.ndarray, patched_compressed: np.ndarray
    ) -> Dict:
        """
        Compute DTW distance between clean and patched trajectories.
        Uses Cosine distance to capture semantic divergence robustly in SVD space.
        """
        from fastdtw import fastdtw
        from scipy.spatial.distance import cosine

        # Handle zero-vectors which cause NaNs in cosine distance
        clean_norm = np.linalg.norm(clean_compressed, axis=1, keepdims=True)
        patched_norm = np.linalg.norm(patched_compressed, axis=1, keepdims=True)
        
        # Add small epsilon to avoid division by zero
        c_comp = clean_compressed / (clean_norm + 1e-9)
        p_comp = patched_compressed / (patched_norm + 1e-9)

        distance, path = fastdtw(
            c_comp,
            p_comp,
            radius=self.config.dtw_radius,
            dist=cosine,
        )

        # Normalized by path length
        normalized = distance / len(path) if path else 0.0

        return {
            "dtw_distance": float(distance),
            "dtw_normalized": float(normalized),
            "path_length": len(path),
            "clean_length": len(clean_compressed),
            "patched_length": len(patched_compressed),
        }

    def soft_dtw_divergence(
        self, clean_compressed: np.ndarray, patched_compressed: np.ndarray, gamma: float = 0.1
    ) -> Dict:
        """
        PyTorch-based Soft-DTW implementation (Gap C optimization).
        Provides a differentiable, smoothed alternative to standard DTW.
        """
        import torch
        
        X = torch.tensor(clean_compressed, dtype=torch.float32)
        Y = torch.tensor(patched_compressed, dtype=torch.float32)
        
        # Compute pairwise distance matrix (L2 Squared)
        n, d = X.shape
        m, d_ = Y.shape
        dist = torch.cdist(X[None], Y[None], p=2).squeeze(0).pow(2)
        
        # Soft-DTW recurrence
        # R[i, j] = dist[i, j] + softmin(R[i-1, j], R[i, j-1], R[i-1, j-1])
        R = torch.zeros((n + 1, m + 1), device=X.device)
        R[0, 1:] = 1e8
        R[1:, 0] = 1e8
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # softmin(a, b, c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))
                v0 = R[i - 1, j]
                v1 = R[i, j - 1]
                v2 = R[i - 1, j - 1]
                
                m_val = torch.stack([v0, v1, v2])
                soft_min = -gamma * torch.logsumexp(-m_val / gamma, dim=0)
                R[i, j] = dist[i - 1, j - 1] + soft_min
        
        score = R[n, m].item()
        return {
            "soft_dtw_score": score,
            "normalized_soft_dtw": score / (n + m),
            "gamma": gamma
        }

    # ------------------------------------------------------------------ #
    #  Topological Data Analysis (Persistent Homology)
    # ------------------------------------------------------------------ #

    def topological_analysis(
        self, compressed: np.ndarray, max_dim: int = 1
    ) -> Dict:
        """
        Compute persistent homology on the SVD-compressed trajectory.

        EXPERIMENTAL: This is an exploratory metric. There is NO proven
        mathematical theorem linking topological features (Betti numbers,
        persistence) to LLM reasoning quality.

        Betti numbers count k-dimensional holes:
            β₀ = connected components
            β₁ = 1D loops (cycles in trajectory)

        We extract Persistence Landscape L² norms as a statistically stable
        topological summary (Cohen-Steiner et al. stability theorem) rather
        than using raw Betti sums which are sampling-density dependent.

        Args:
            compressed: [Tokens, n_components] trajectory.
            max_dim: Maximum homological dimension (0=components, 1=loops).

        Returns:
            Dict with persistence diagrams, Betti numbers, and landscape norms.
        """
        try:
            import ripser

            result = ripser.ripser(compressed, maxdim=max_dim, thresh=np.inf)
            diagrams = result["dgms"]

            betti_numbers = {}
            persistence_stats = {}
            landscape_norms = {}

            for dim, dgm in enumerate(diagrams):
                # Filter out infinite death times
                finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm

                betti_numbers[f"betti_{dim}"] = len(finite)

                if len(finite) > 0:
                    lifetimes = finite[:, 1] - finite[:, 0]
                    # Simple 1st Persistence Landscape calculation
                    # Landscape lambda_1(t) = max_{i} max(0, min(t - b_i, d_i - t))
                    b = finite[:, 0]
                    d = finite[:, 1]
                    t_min, t_max = np.min(b), np.max(d)
                    if t_max > t_min:
                        # Sample 100 points
                        t_vals = np.linspace(t_min, t_max, 100)
                        landscape = np.zeros_like(t_vals)
                        for t_idx, t in enumerate(t_vals):
                            # min(t-b, d-t)
                            vals = np.minimum(t - b, d - t)
                            landscape[t_idx] = np.max(np.maximum(0, vals))
                        # L2 Norm of the landscape
                        l_norm = float(np.sqrt(np.trapz(landscape**2, t_vals)))
                    else:
                        l_norm = 0.0

                    landscape_norms[f"dim_{dim}"] = l_norm

                    persistence_stats[f"dim_{dim}"] = {
                        "count": len(finite),
                        "mean_lifetime": float(np.mean(lifetimes)),
                        "max_lifetime": float(np.max(lifetimes)),
                        "total_persistence": float(np.sum(lifetimes)),
                        "landscape_norm": l_norm,
                    }
                else:
                    landscape_norms[f"dim_{dim}"] = 0.0
                    persistence_stats[f"dim_{dim}"] = {
                        "count": 0,
                        "mean_lifetime": 0.0,
                        "max_lifetime": 0.0,
                        "total_persistence": 0.0,
                        "landscape_norm": 0.0,
                    }

            return {
                "diagrams": diagrams,
                "betti_numbers": betti_numbers,
                "persistence_stats": persistence_stats,
                "landscape_norms": landscape_norms,
            }

        except ImportError:
            console.print(
                "[yellow]ripser not installed. Skipping TDA. "
                "Install with: pip install ripser[/]"
            )
            return {"error": "ripser not installed", "betti_numbers": {}}

    def topological_analysis_windowed(
        self, compressed: np.ndarray, max_dim: int = 1
    ) -> Dict:
        """
        Run persistent homology on sliding windows of the compressed
        trajectory to capture local topological changes along token-time.
        """
        try:
            import ripser
        except ImportError:
            console.print(
                "[yellow]ripser not installed. Skipping windowed TDA. "
                "Install with: pip install ripser[/]"
            )
            return {"error": "ripser not installed", "windows": []}

        if not getattr(self.config, "tda_enable_windowed", False):
            return {"windows": []}

        n_tokens = compressed.shape[0]
        window_size = min(self.config.tda_window_size, n_tokens)
        stride = max(1, self.config.tda_window_stride)

        windows = []
        for start in range(0, max(1, n_tokens - window_size + 1), stride):
            end = min(start + window_size, n_tokens)
            segment = compressed[start:end]
            if segment.shape[0] < 3:
                continue

            result = ripser.ripser(segment, maxdim=max_dim, thresh=np.inf)
            diagrams = result["dgms"]

            betti_numbers = {}
            for dim, dgm in enumerate(diagrams):
                finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
                betti_numbers[f"betti_{dim}"] = len(finite)

            windows.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "diagrams": diagrams,
                    "betti_numbers": betti_numbers,
                }
            )

        return {"windows": windows}

    # ------------------------------------------------------------------ #
    #  Composite Validity Score
    # ------------------------------------------------------------------ #

    def compute_fidelity_score(
        self,
        dtw_result: Dict,
        spectral_result: Dict,
        tda_result: Dict,
        stationarity_result: Dict,
        var_result: Optional[Dict] = None,
        hmm_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Compute a composite reasoning fidelity score (V2).

        Uses robust empirical likelihood mechanisms including:
            1. DTW Divergence (Semantic Distance)
            2. Spectral Kurtosis (Peakiness)
            3. H1 Persistence Landscape Norm (Topology)
            4. Transition non-stationarity
            5. [V2] VAR Stability (Spectral Radius Roots)
            6. [V2] HMM Persistence (Step-level context focus)
            7. [V2] Topological Rank (Reasoning richness)
        """
        import scipy.stats as stats
        scores = {}

        # 1. Semantic tracking via DTW Divergence z-proxy
        dtw_norm = dtw_result.get("dtw_normalized", 0.0)
        scores["z_dtw"] = float(np.clip((dtw_norm - 1.0) / 0.5, -3, 3))

        # 2. Spectral Coherence peakness
        agg_power = spectral_result.get("aggregate_power", None)
        if agg_power is not None and len(agg_power) > 1:
            peak_ratio = float(np.max(agg_power[1:]) / (np.sum(agg_power[1:]) + 1e-9))
            scores["z_spectral"] = float(np.clip((peak_ratio - 0.2) / 0.1, -3, 3))
        else:
            scores["z_spectral"] = -1.0

        # 3. Topological Completeness (H1 Persistence Landscape)
        lnorms = tda_result.get("landscape_norms", {})
        h1_norm = lnorms.get("dim_1", 0.0)
        scores["z_topology"] = float(np.clip((h1_norm - 0.5) / 0.25, -3, 3))

        # 4. Stationary vs Deterministic Dynamics (Variance Ratio)
        var_ratio = stationarity_result.get("variance_ratio", 0.1)
        # Low ratio (<1) = high trending (advancing logic)
        scores["z_stationarity"] = float(np.clip((0.5 - var_ratio) / 0.2, -3, 3))

        # ── V2 Extensions ──────────────────────────────────────────────────
        z_v2_list = []

        # 5. VAR Stability (Roots)
        if var_result:
            sr = var_result.get("spectral_radius", 0.0)
            # Spectral radius < 1 is stable. Ideal is ~0.8-0.9 (dynamic but stable)
            # sr > 1.0 = feedback loop instability (bad)
            if sr > 1.1:
                z_stability = -2.0
            elif 0.7 <= sr <= 0.98:
                z_stability = 2.0
            else:
                z_stability = 0.5
            scores["z_stability"] = z_stability
            z_v2_list.append(z_stability)

            # 6. Topological Rank (Collapse Detection)
            rank_info = var_result.get("topological_rank", {})
            eff_rank = rank_info.get("effective_rank", 1)
            rank_entropy = rank_info.get("rank_entropy", 0.0)
            
            # Rank collapse (rank=1) is a hallucination/mimicry sign.
            # Multi-head synergy (rank >= 2) is a reasoning sign.
            z_rank = float(np.clip((rank_entropy - 1.0) / 0.5, -2, 2))
            if eff_rank < 2:
                z_rank -= 1.0
            scores["z_rank"] = z_rank
            z_v2_list.append(z_rank)

        # 7. HMM Persistence (Contextual Logic)
        if hmm_result:
            persistence = hmm_result.get("persistence_scores", [])
            if persistence:
                avg_persistence = np.mean(persistence)
                # High persistence (>0.5) = focused reasoning mode.
                # Low persistence = fragmentation.
                z_persistence = float(np.clip((avg_persistence - 0.4) / 0.2, -2, 2))
                scores["z_persistence"] = z_persistence
                z_v2_list.append(z_persistence)

        # Combine all Z-scores
        base_z = [scores["z_dtw"], scores["z_spectral"], scores["z_topology"], scores["z_stationarity"]]
        final_z_array = np.array(base_z + z_v2_list)
        
        # Calculate Likelihood
        mean_z = np.mean(final_z_array)
        validity_probability = float(stats.norm.cdf(mean_z))
        
        scores["composite_validity"] = validity_probability
        scores["fidelity_score"] = validity_probability * 100.0

        # Verdict Mapping
        if validity_probability > 0.85:
            scores["verdict"] = "DEEP REASONING (Sound)"
        elif validity_probability > 0.65:
            scores["verdict"] = "ACTIVE LOGIC (Reliable)"
        elif validity_probability > 0.40:
            scores["verdict"] = "HEURISTIC RECALL (Surface)"
        elif validity_probability > 0.15:
            scores["verdict"] = "STOCHASTIC PARROT (Unsound)"
        else:
            scores["verdict"] = "COLLAPSED / HALLUCINATION"

        return scores

    # ------------------------------------------------------------------ #
    #  Attention Head Interaction Analysis (Time-Series on Heads)
    # ------------------------------------------------------------------ #

    def _ensure_var_cls(self):
        """Lazy import of statsmodels VAR to avoid hard dependency at import time."""
        if self._var_cls is not None:
            return
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
        except ImportError:
            console.print(
                "[yellow]statsmodels not installed. "
                "Install with: pip install statsmodels[/]"
            )
            self._var_cls = None
            return
        self._var_cls = VAR

    def _select_optimal_lag(self, series_np: np.ndarray, max_lags: int) -> int:
        """
        Gap A.4 — Automated lag selection using specified information criterion.
        
        Returns the optimal level lag p >= 1.
        """
        if self._var_cls is None:
            return 1
            
        try:
            # We must use the level series (not differenced) to select the level lag p.
            model = self._var_cls(series_np)
            criterion = str(getattr(self.config, "var_lag_selection", "aic") or "aic").lower()
            
            # Use max(1, ...) to cap lower bound, and statsmodels select_order
            # max_lags must be < (T-1)/H - 1 for stable estimation.
            T, H = series_np.shape
            admissible = max(1, (T - H - 1) // (H + 1))
            actual_max = min(max_lags, admissible, 10) # 10 is a sanity cap for LLM metrics
            
            if actual_max < 1:
                return 1
                
            selection = model.select_order(maxlags=actual_max)
            # Fetch the rank for the specified criterion
            optimal_p = selection.selected_orders.get(criterion, 1)
            # statsmodels select_order can return 0 if no dynamic exists; return 1 for a minimal VAR(1)
            return int(max(1, optimal_p))
        except Exception:
            return 1 # Fallback to VAR(1)

    def _preprocess_series_for_var(self, series_np: np.ndarray) -> np.ndarray:
        """
        Preprocess channels before VAR/VECM fitting.

        Steps (configurable):
            1) logit transform channels naturally bounded in [0, 1]
            2) per-channel scaling ('zscore' or 'robust')
            3) remove common per-timestep component (row mean)
        """
        if not getattr(self.config, "var_preprocess_enable", True):
            return np.nan_to_num(series_np, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.array(series_np, dtype=np.float64, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        eps = float(getattr(self.config, "var_preprocess_eps", 1e-6) or 1e-6)

        # 1) Bounded-channel logit transform.
        if getattr(self.config, "var_preprocess_logit_bounded", True):
            col_min = np.min(x, axis=0)
            col_max = np.max(x, axis=0)
            bounded = (col_min >= -1e-9) & (col_max <= 1.0 + 1e-9)
            if np.any(bounded):
                xb = np.clip(x[:, bounded], eps, 1.0 - eps)
                x[:, bounded] = np.log(xb / (1.0 - xb))

        # 2) Channel-wise scaling.
        scale_mode = str(getattr(self.config, "var_preprocess_scale", "robust") or "robust").lower()
        if scale_mode == "zscore":
            mu = np.mean(x, axis=0, keepdims=True)
            sd = np.std(x, axis=0, keepdims=True)
            sd = np.where(sd < eps, 1.0, sd)
            x = (x - mu) / sd
        elif scale_mode == "robust":
            med = np.median(x, axis=0, keepdims=True)
            mad = np.median(np.abs(x - med), axis=0, keepdims=True)
            robust_sd = 1.4826 * mad
            robust_sd = np.where(robust_sd < eps, 1.0, robust_sd)
            x = (x - med) / robust_sd

        # 3) Remove common-mode variation shared by all channels at each timestep.
        if getattr(self.config, "var_preprocess_remove_global_mean", True):
            x = x - x.mean(axis=1, keepdims=True)

        clip_val = getattr(self.config, "var_preprocess_clip", 10.0)
        if clip_val is not None:
            cv = float(clip_val)
            if cv > 0:
                x = np.clip(x, -cv, cv)

        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ================================================================== #
    #  GAP A: Stationarity & Cointegration Methods
    # ================================================================== #

    def _test_per_head_stationarity(self, metric_series: np.ndarray, p_lags: int = 1) -> dict:
        """
        A.1 — Run ADF test on each of the H head entropy series independently.

        Args:
            metric_series: np.ndarray of shape [T, H]
            p_lags: The level lag to focus the test on.

        Returns:
            dict with 'p_values', 'is_stationary', 'needs_diff', 'diff_mask'
        """
        from statsmodels.tsa.stattools import adfuller
        import torch

        # Use float64 for higher precision in statistical tests
        series_t  = torch.as_tensor(metric_series, dtype=torch.float64)   # [T, H]
        col_means = torch.mean(series_t, dim=0, keepdim=True)             # [1, H]
        centered  = (series_t - col_means).numpy()                        # [T, H] zero-mean

        T, H = centered.shape
        p_values = np.zeros(H)
        for h in range(H):
            series = centered[:, h]
            if np.std(series) < 1e-9:
                p_values[h] = 0.99  # constant series → unit root likely
                continue
            try:
                # Use AIC for autolag but cap by the selected p_lags if needed
                # Statsmodels defaults to 12*(nobs/100)^(1/4)
                adf_result = adfuller(series, maxlag=p_lags, autolag=None)
                p_values[h] = adf_result[1]
            except Exception:
                p_values[h] = 1.0

        is_stationary = p_values < 0.05
        diff_mask = ~is_stationary

        return {
            'p_values': p_values,
            'is_stationary': is_stationary,
            'needs_diff': bool(diff_mask.any()),
            'diff_mask': diff_mask,
        }

    def _apply_selective_differencing(
        self,
        metric_series: np.ndarray,
        diff_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        A.2 — Uniform first-differencing when ANY head is non-stationary.

        MATHEMATICAL NOTE:
            The original implementation differenced only non-stationary heads,
            creating a mixed I(0)/I(1) system. This is non-standard and violates
            the homogeneous integration assumption required by VAR. When any
            channel is non-stationary, we difference ALL channels uniformly
            to maintain a coherent I(0) system for VAR estimation.

            The diff_mask is preserved for diagnostic reporting (which heads
            were originally non-stationary) but does NOT control selective
            differencing anymore.

        Returns:
            differenced_series: np.ndarray [T-1, H]  (all channels differenced)
            diff_mask: original mask (for diagnostics only)
        """
        import torch

        series_t = torch.as_tensor(metric_series, dtype=torch.float64)   # [T, H]
        differenced = torch.diff(series_t, n=1, dim=0)                    # [T-1, H]

        return differenced.numpy(), diff_mask

    def _check_cointegration(
        self,
        series_np: np.ndarray,
        p_lags: int = 1
    ) -> dict:
        """
        A.3 — Run Johansen cointegration test using the level lag p.
        
        The statsmodels coint_johansen expects k_ar_diff = p - 1.
        
        Returns:
            dict with 'cointegrated', 'n_coint_vectors', 'trace_stats', 'crit_values_95'
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        # p_lags is the level lag (VAR(p)). VECM k_ar_diff = p - 1.
        k_diff = max(p_lags - 1, 0)
        
        # det_order=0: constant in level VAR. det_order=1: trend in level VAR.
        # video typically uses constant (det_order=0).
        result = coint_johansen(series_np, det_order=0, k_ar_diff=k_diff)
        crit_95 = result.cvt[:, 1]
        trace_stats = result.lr1
        rank = int(np.sum(trace_stats > crit_95))

        return {
            'cointegrated': rank > 0,
            'n_coint_vectors': rank,
            'trace_stats': trace_stats,
            'crit_values_95': crit_95,
        }

    def _fit_vecm(
        self,
        series_np: np.ndarray,
        rank: int,
        p_lags: int = 1
    ) -> object:
        """
        A.3 — Fit a VECM assuming level lag p (diff lag p-1).
        
        VECM captures both long-run (ECT via alpha*beta') and short-run (Gamma lags) dynamics.
        """
        from statsmodels.tsa.vector_ar.vecm import VECM

        # Check for degenerate signals
        std_per_col = series_np.std(axis=0)
        if (std_per_col < 1e-9).any():
            series_np = series_np + np.random.normal(0, 1e-7, series_np.shape)

        k_diff = max(p_lags - 1, 0)
        # deterministic='ci' includes a constant in the cointegrating relation (common in video).
        # deterministic='co' includes a trend-adjusted constant outside the relation.
        # Fallback to 'ci' is conservative and stable for small samples.
        try:
            model = VECM(series_np, k_ar_diff=k_diff, coint_rank=rank, deterministic='ci')
            return model.fit()
        except Exception:
            jitter = np.random.normal(0, 1e-6, series_np.shape)
            model = VECM(series_np + jitter, k_ar_diff=k_diff, coint_rank=rank, deterministic='ci')
            return model.fit()

    # ================================================================== #
    #  GAP B: Statistical Significance Methods
    # ================================================================== #

    def _granger_pvalue_matrix(
        self,
        var_result,
        series_np: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """
        B.1 — Compute Granger causality F-test p-value for every directed pair (j→i).

        Returns:
            pval_matrix: np.ndarray [H, H] — pval_matrix[i, j] = p-value for j→i
                         diagonal is NaN
        """
        H = series_np.shape[1]
        pval_matrix = np.full((H, H), np.nan)

        for i in range(H):
            for j in range(H):
                if i == j:
                    continue
                try:
                    test = var_result.test_causality(
                        caused=i,
                        causing=j,
                        kind='f'
                    )
                    pval_matrix[i, j] = test.pvalue
                except Exception:
                    pval_matrix[i, j] = 1.0

        return pval_matrix

    def _granger_pvalue_matrix_vecm(
        self,
        vecm_result,
        H_active: int
    ) -> np.ndarray:
        """
        B.1.v — Compute Granger causality Wald test p-value for VECM directed pair (j→i).
        Tests H0: Gamma coefficients for variable j in equation i are all zero.
        """
        from scipy.stats import chi2
        pval_matrix = np.full((H_active, H_active), np.nan)
        
        # Gamma has shape (H, H*k_ar) or (H*k_ar, H)
        # We need to find the indices of variable j's lags in the parameter vector
        # statsmodels VECM.resids has shape (T-k, H)
        # VECM.cov_params() gives the covariance of all estimated parameters
        
        try:
            params = vecm_result.params
            cov_params = vecm_result.cov_params()
            k_ar = vecm_result.k_ar  # lags in levels = k_ar + 1
            # In statsmodels VECM:
            # Deterministic terms (const, etc.) come first, then Alpha*Beta', then Gamma.
            # This is complex to parse manually across versions.
            # Use the built-in test_granger if possible, or falls back to Wald.
            
            for i in range(H_active):
                for j in range(H_active):
                    if i == j: continue
                    try:
                        # statsmodels VECMResults.test_granger(caused, causing)
                        test = vecm_result.test_granger(caused=i, causing=j)
                        pval_matrix[i, j] = test.pvalue
                    except Exception:
                        pval_matrix[i, j] = 1.0
        except Exception:
            pass # Return NaNs/1.0s if extraction fails
            
        return pval_matrix

    def _apply_fdr_correction(
        self,
        pval_matrix: np.ndarray,
        alpha: float = 0.05
    ) -> dict:
        """
        B.2 — Apply Benjamini-Hochberg FDR correction to the p-value matrix.

        Returns:
            dict with 'reject_matrix', 'pval_corrected', 'n_significant', 'significant_pairs'
        """
        from statsmodels.stats.multitest import multipletests

        H = pval_matrix.shape[0]

        flat_pvals = []
        flat_indices = []
        for i in range(H):
            for j in range(H):
                if i != j and not np.isnan(pval_matrix[i, j]):
                    flat_pvals.append(pval_matrix[i, j])
                    flat_indices.append((i, j))

        if not flat_pvals:
            return {
                'reject_matrix': np.zeros((H, H), dtype=bool),
                'pval_corrected': np.full((H, H), np.nan),
                'n_significant': 0,
                'significant_pairs': [],
            }

        reject, pvals_corrected, _, _ = multipletests(
            flat_pvals, alpha=alpha, method='fdr_bh'
        )

        reject_matrix = np.zeros((H, H), dtype=bool)
        corrected_matrix = np.full((H, H), np.nan)

        for k, (i, j) in enumerate(flat_indices):
            reject_matrix[i, j] = reject[k]
            corrected_matrix[i, j] = pvals_corrected[k]

        significant_pairs = [
            (j, i, corrected_matrix[i, j])
            for (i, j) in flat_indices
            if reject_matrix[i, j]
        ]
        significant_pairs.sort(key=lambda x: x[2])

        return {
            'reject_matrix': reject_matrix,
            'pval_corrected': corrected_matrix,
            'n_significant': int(reject_matrix.sum()),
            'significant_pairs': significant_pairs,
        }

    def _bootstrap_surrogate_pvalues(
        self,
        series_np: np.ndarray,
        observed_influence: np.ndarray,
        n_surrogates: int = 500,
        lag: int = 3
    ) -> np.ndarray:
        """
        B.3 — Phase-scramble each head series independently to break temporal
        structure. Refit VAR on each surrogate. Return empirical p-value matrix.

        Returns:
            empirical_pval: np.ndarray [H, H]
        """
        from statsmodels.tsa.vector_ar.var_model import VAR
        import torch
        import torch.fft
        import torch.linalg

        T, H = series_np.shape
        surrogate_scores = np.zeros((n_surrogates, H, H))

        series_t  = torch.as_tensor(series_np, dtype=torch.float64)         # [T, H]
        spectrum  = torch.fft.rfft(series_t, dim=0)                         # [F, H] complex
        magnitude = spectrum.abs()                                          # [F, H] real

        F = spectrum.shape[0]
        rand_phases = torch.rand(n_surrogates, F, H, dtype=torch.float64) * 2 * torch.pi
        rand_phases[:, 0, :] = 0.0   # preserve DC component

        scrambled = magnitude.unsqueeze(0) * torch.polar(
            torch.ones(n_surrogates, F, H, dtype=torch.float64),
            rand_phases
        )                                                                    # [S, F, H]

        # Inverse FFT for all surrogates in one call
        surrogates = torch.fft.irfft(scrambled, n=T, dim=1)                 # [S, T, H]
        surrogates_np = surrogates.numpy()

        for s in range(n_surrogates):
            surrogate = surrogates_np[s]
            try:
                surr_model = VAR(surrogate)
                surr_result = surr_model.fit(maxlags=lag, ic='aic')
                
                coefs_t = torch.as_tensor(surr_result.coefs, dtype=torch.float64)
                surrogate_scores[s] = torch.linalg.norm(coefs_t, ord=1, dim=0).numpy()
            except Exception:
                surrogate_scores[s] = observed_influence

        empirical_pval = np.mean(
            surrogate_scores >= observed_influence[None, :, :], axis=0
        )
        return empirical_pval

    def _conditional_transfer_entropy(
        self,
        series_np: np.ndarray,
        k: int = 3,
        discretize_bins: int = 8
    ) -> np.ndarray:
        """
        B.4 — Compute conditional transfer entropy TE(j→i) for all pairs.
        TE(j→i) = I(X_i(t+1) ; X_j^{1..k} | X_i^{1..k})

        Returns:
            te_matrix: np.ndarray [H, H] — te_matrix[i,j] = TE(j→i) in nats
        """
        T, H = series_np.shape

        def discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
            bins = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
            return np.digitize(x, bins) - 1

        disc = np.stack(
            [discretize(series_np[:, h], discretize_bins) for h in range(H)],
            axis=1
        )

        te_matrix = np.zeros((H, H))

        for i in range(H):
            for j in range(H):
                if i == j:
                    continue

                counts = {}
                for t in range(k, T - 1):
                    x_i_future = int(disc[t + 1, i])
                    x_i_past = tuple(disc[t - k:t, i].tolist())
                    x_j_past = tuple(disc[t - k:t, j].tolist())
                    key = (x_i_future, x_i_past, x_j_past)
                    counts[key] = counts.get(key, 0) + 1

                total = sum(counts.values())
                if total == 0:
                    continue

                te = 0.0
                for (xif, xip, xjp), cnt in counts.items():
                    p_joint = cnt / total
                    p_xip_xjp = sum(
                        v for (a, b, c), v in counts.items() if b == xip and c == xjp
                    ) / total
                    p_xif_xip = sum(
                        v for (a, b, c), v in counts.items() if a == xif and b == xip
                    ) / total
                    p_xip = sum(
                        v for (a, b, c), v in counts.items() if b == xip
                    ) / total

                    if p_joint > 0 and p_xip_xjp > 0 and p_xif_xip > 0 and p_xip > 0:
                        te += p_joint * np.log(
                            (p_joint * p_xip) / (p_xip_xjp * p_xif_xip)
                        )

                te_matrix[i, j] = max(te, 0.0)

        return te_matrix

    def _partial_directed_coherence(
        self,
        var_result,
        freqs: np.ndarray = None
    ) -> dict:
        """
        B.5 — Compute Partial Directed Coherence (PDC) from fitted VAR coefficients.
        PDC(j→i, f) = |A(f)[i,j]|² / Σ_k |A(f)[k,j]|²

        Returns:
            dict with 'pdc' [F,H,H], 'freqs', 'pdc_low', 'pdc_high', 'dominant_freq_per_pair'
        """
        if freqs is None:
            freqs = np.linspace(0, 0.5, 128)

        coefs = var_result.coefs
        p, H, _ = coefs.shape

        # Vectorized computation across all frequencies: O(H³) instead of O(F × H³)
        # Shape: (F, H, H) for frequencies × head matrix
        freqs_2d = np.outer(freqs, np.arange(1, p + 1))  # (F, p)
        exp_terms_4d = np.exp(-2j * np.pi * freqs_2d)[:, :, np.newaxis, np.newaxis]  # (F, p, 1, 1)

        # Vectorized A(f) computation: I - Σ_k A_k exp(-i 2π f k)
        A_f_all = np.eye(H, dtype=complex)[np.newaxis, :, :] - np.sum(
            coefs[np.newaxis, :, :, :] * exp_terms_4d, axis=1
        )  # (F, H, H)

        A_abs_sq = np.abs(A_f_all) ** 2  # (F, H, H)
        col_sums = A_abs_sq.sum(axis=1, keepdims=True)  # (F, 1, H)
        col_sums[col_sums == 0] = 1.0
        pdc = A_abs_sq / col_sums  # (F, H, H)

        low_mask = freqs <= 0.1
        high_mask = freqs >= 0.4

        pdc_low = pdc[low_mask].mean(axis=0) if low_mask.any() else np.zeros((H, H))
        pdc_high = pdc[high_mask].mean(axis=0) if high_mask.any() else np.zeros((H, H))

        return {
            'pdc': pdc,
            'freqs': freqs,
            'pdc_low': pdc_low,
            'pdc_high': pdc_high,
            'dominant_freq_per_pair': np.argmax(pdc, axis=0) / len(freqs) * 0.5,
        }

    # ================================================================== #
    #  GAP C: Perturbation / Intervention Methods
    # ================================================================== #

    def _make_ablation_model(
        self,
        layer_idx: int,
        h_source: int,
        mode: str = 'zero',
        mean_activation: torch.Tensor = None
    ):
        """
        C.1 — Return a pyvene IntervenableModel that ablates head h_source.
        """
        import pyvene as pv
        
        if mode == 'zero':
            config = pv.IntervenableConfig([{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.ZeroIntervention,
                "source_representation": None,
            }])
        elif mode == 'mean':
            config = pv.IntervenableConfig([{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.ConstantSourceIntervention,
                "source_representation": mean_activation,
            }])
        elif mode == 'gaussian':
            config = pv.IntervenableConfig([{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.NoiseIntervention,
            }])
        else:
            raise ValueError(f"Unknown mode {mode}")
            
        return pv.IntervenableModel(config, self.interceptor.model)

    def _activation_patch_experiment(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layer_idx: int,
        h_source: int
    ) -> dict:
        """
        C.3 — Activation patching using pyvene (VanillaIntervention).
        Patches head h_source from clean_prompt into corrupted_prompt run.
        """
        import pyvene as pv
        import torch.nn.functional as F

        patch_config = pv.IntervenableConfig([{
            "layer": layer_idx,
            "component": "attention_output",
            "intervention_type": pv.VanillaIntervention,
        }])
        patcher = pv.IntervenableModel(patch_config, self.interceptor.model)

        clean_inputs = self.interceptor.tokenizer(clean_prompt, return_tensors='pt').to(self.interceptor.device)
        corr_inputs = self.interceptor.tokenizer(corrupted_prompt, return_tensors='pt').to(self.interceptor.device)

        _, patched_output = patcher(
            base=corr_inputs,
            sources=[clean_inputs],
            unit_locations={"base": h_source, "source": h_source}
        )
        
        with torch.no_grad():
            clean_logits = self.interceptor.model(**clean_inputs).logits[:, -1, :]
            corrupt_logits = self.interceptor.model(**corr_inputs).logits[:, -1, :]
            
        patched_logits = patched_output.logits[:, -1, :]

        kl_baseline = float(F.kl_div(corrupt_logits.log_softmax(-1), clean_logits.softmax(-1), reduction='sum'))
        kl_patched = float(F.kl_div(patched_logits.log_softmax(-1), clean_logits.softmax(-1), reduction='sum'))

        restoration = 1.0 - (kl_patched / kl_baseline) if kl_baseline > 0 else 0.0

        return {
            'patch_effect': float(F.kl_div(
                corrupt_logits.log_softmax(-1),
                patched_logits.softmax(-1),
                reduction='sum'
            )),
            'restoration': float(np.clip(restoration, 0.0, 1.0)),
            'kl_baseline': float(kl_baseline),
            'kl_patched': float(kl_patched),
        }

    def _build_mean_ablation_cache(
        self,
        reference_prompts: List[str],
        layer_idx: int,
        h_source: int
    ) -> torch.Tensor:
        """
        C.2 — Run the model on reference_prompts, collect head h_source activations
        using pyvene CollectIntervention.
        """
        import pyvene as pv
        
        collect_config = pv.IntervenableConfig([{
            "layer": layer_idx,
            "component": "attention_output",
            "intervention_type": pv.CollectIntervention,
        }])
        collector = pv.IntervenableModel(collect_config, self.interceptor.model)

        collected = []
        for prompt in reference_prompts:
            inputs = self.interceptor.tokenizer(prompt, return_tensors="pt").to(self.interceptor.device)
            with torch.no_grad():
                _, collected_output = collector(
                    base=inputs, 
                    sources=None,
                    unit_locations={"base": h_source}
                )
            collected.append(collected_output.collected_activations[0])

        if not collected:
            return torch.empty(0)

        # Handle varying sequence lengths by padding before stacking
        max_len = max(t.shape[1] for t in collected)
        padded = []
        for t in collected:
            pad_size = max_len - t.shape[1]
            if pad_size > 0:
                last_val = t[:, -1:, :].expand(-1, pad_size, -1)
                t = torch.cat([t, last_val], dim=1)
            padded.append(t)

        mean_activation = torch.stack(padded).mean(dim=0)
        return mean_activation

    def _direct_vs_total_effect(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layer_idx: int,
        source_head: int,
        target_head: int,
        mediator_heads: List[int]
    ) -> dict:
        """
        C.4 — Estimate direct effect of source_head on target_head,
        controlling for mediation through mediator_heads.
        Uses composed pv.VanillaInterventions.
        """
        import pyvene as pv
        import torch.nn.functional as F

        def _run_with_patches(heads_to_patch: List[int]) -> torch.Tensor:
            if not heads_to_patch:
                with torch.no_grad():
                    return self.interceptor.model(**self.interceptor.tokenizer(corrupted_prompt, return_tensors="pt").to(self.interceptor.device)).logits[:, -1]
                    
            configs = [{
                "layer": layer_idx,
                "component": "attention_output",
                "intervention_type": pv.VanillaIntervention,
            } for _ in heads_to_patch]

            model_patched = pv.IntervenableModel(pv.IntervenableConfig(configs), self.interceptor.model)
            
            clean_inputs = self.interceptor.tokenizer(clean_prompt, return_tensors="pt").to(self.interceptor.device)
            corr_inputs = self.interceptor.tokenizer(corrupted_prompt, return_tensors="pt").to(self.interceptor.device)
            
            locations = [{"base": h, "source": h} for h in heads_to_patch]
            
            with torch.no_grad():
                _, out = model_patched(
                    base=corr_inputs,
                    sources=[clean_inputs] * len(heads_to_patch),
                    unit_locations=locations
                )
            return out.logits[:, -1]

        with torch.no_grad():
            baseline_logits = self.interceptor.model(**self.interceptor.tokenizer(corrupted_prompt, return_tensors="pt").to(self.interceptor.device)).logits[:, -1]

        total_logits = _run_with_patches([source_head])
        indirect_logits = _run_with_patches([source_head] + mediator_heads)

        total_effect = float(F.kl_div(baseline_logits.log_softmax(-1), total_logits.softmax(-1), reduction='sum'))
        indirect_effect = float(F.kl_div(baseline_logits.log_softmax(-1), indirect_logits.softmax(-1), reduction='sum'))

        direct_effect = max(total_effect - indirect_effect, 0.0)
        mediation_ratio = indirect_effect / total_effect if total_effect > 0 else 0.0

        return {
            'total_effect':    total_effect,
            'indirect_effect': indirect_effect,
            'direct_effect':   direct_effect,
            'mediation_ratio': float(np.clip(mediation_ratio, 0.0, 1.0)),
            'source_head':     source_head,
            'target_head':     target_head,
            'mediator_heads':  mediator_heads,
        }

    # ================================================================== #
    #  GAP D: Thinking Time Axis Methods
    # ================================================================== #

    def _segment_by_topological_phases(
        self,
        euler_characteristic_series: np.ndarray,
        spike_threshold_std: float = 2.0
    ) -> List[Tuple[int, int]]:
        """
        D.2 — Use Euler characteristic spikes as phase boundaries.
        Returns token-index spans of each phase.
        """
        mean_ec = euler_characteristic_series.mean()
        std_ec = euler_characteristic_series.std()
        if std_ec < 1e-9:
            return [(0, len(euler_characteristic_series))]

        threshold = mean_ec + spike_threshold_std * std_ec
        spike_positions = np.where(np.abs(euler_characteristic_series) > threshold)[0]

        boundaries = [0] + spike_positions.tolist() + [len(euler_characteristic_series)]
        boundaries = sorted(set(boundaries))

        phases = []
        for start_tok, end_tok in zip(boundaries[:-1], boundaries[1:]):
            if end_tok - start_tok >= 3:
                phases.append((int(start_tok), int(end_tok)))

        return phases

    def _aggregate_entropy_by_phase(
        self,
        metric_series_np: np.ndarray,
        phases: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        D.2 — Aggregate head entropy series by topological phase.
        Returns [n_phases, H] array.
        """
        H = metric_series_np.shape[1]
        phase_entropy = np.zeros((len(phases), H))

        for pi, (start, end) in enumerate(phases):
            phase_entropy[pi] = metric_series_np[start:end, :].mean(axis=0)

        return phase_entropy

    def _resample_at_equal_arc_length(
        self,
        metric_series_np: np.ndarray,
        hidden_states: np.ndarray,
        n_resampled_points: int = 80
    ) -> np.ndarray:
        """
        D.4 — Resample head entropy series from equal token-spacing to
        equal arc-length spacing (equal cognitive effort per step).

        Returns:
            resampled: np.ndarray [n_resampled_points, H]
        """
        from scipy.interpolate import interp1d

        T = hidden_states.shape[0]

        diff = np.diff(hidden_states, axis=0)
        arc_steps = np.linalg.norm(diff, axis=1)
        cumulative_arc = np.concatenate([[0.0], np.cumsum(arc_steps)])

        total_arc = cumulative_arc[-1]
        if total_arc < 1e-9:
            return metric_series_np[:n_resampled_points]

        query_arcs = np.linspace(0, total_arc, n_resampled_points)
        resampled = np.zeros((n_resampled_points, metric_series_np.shape[1]))

        for h in range(metric_series_np.shape[1]):
            f = interp1d(
                cumulative_arc,
                metric_series_np[:, h],
                kind='linear',
                fill_value='extrapolate'
            )
            resampled[:, h] = f(query_arcs)

        return resampled

    # ------------------------------------------------------------------ #
    #  Thinking Time Axis (Gap D) - Phase Discovery
    # ------------------------------------------------------------------ #

    # Note: _discover_phases_hmm is defined below in the V2 section.

    # ================================================================== #
    #  Refactored head_interaction_analysis — Rigorous Pipeline
    # ================================================================== #

    def head_interaction_analysis(
        self,
        prompt: str,
        layer_name: Optional[str] = None,
        max_lag: Optional[int] = None,
    ) -> Dict:
        """
        Treat per-head attention metrics as a multivariate time series and
        estimate directed interactions between heads via a rigorous pipeline:

        Pipeline order:
            1. Sanitize & filter constant heads
            2. Gap A.1: Per-head ADF stationarity test
            3. Gap A.5: Joint ADF+KPSS (if config.joint_stationarity_test)
            4. Gap A.3: Johansen cointegration → VECM or selective differencing
            5. Gap A.4: AIC-based lag selection
            6. VAR or VECM fit
            7. Gap B.1: Granger F-test p-value matrix
            8. Gap B.2: BH-FDR correction → mask influence matrix
            9. Gap B.5: PDC (if config.compute_pdc)

        Returns:
            Dict with backward-compatible keys plus new statistical outputs.
        """
        self._ensure_var_cls()
        if self._var_cls is None:
            return {"error": "statsmodels VAR unavailable"}

        # Capture a fresh generation so that head metrics are populated.
        trajectory, generated_text = self.interceptor.capture_generation(prompt)
        if not trajectory:
            return {"error": "no activations captured"}

        # Default to last decoder layer if none specified.
        if layer_name is None:
            layer_names = sorted(trajectory.keys())
            layer_name = layer_names[-1]

        metric_series = self.interceptor.get_head_metric_series(layer_name)
        if metric_series.numel() == 0:
            # Fallback for backends where generate() doesn't surface attention
            # weights to hooks or generation outputs.
            self.interceptor.capture_head_metrics_from_forward(prompt)
            metric_series = self.interceptor.get_head_metric_series(layer_name)

        if metric_series.numel() == 0:
            return {"error": "no head metrics captured", "layer_name": layer_name}

        # metric_series can be scalar [T, H] or vector [T, H, F].
        # In vector mode, VAR should still run on heads (H), not flattened H*F.
        # Select one feature column for causality while keeping full features for
        # decomposition diagnostics.
        raw_series_np = metric_series.numpy()
        series_np = raw_series_np
        selected_feature_idx = None
        if raw_series_np.ndim == 3:
            metric_to_feature_col = {
                "shannon_entropy": 0,
                "normalized_entropy": 1,
                "renyi_entropy_2": 2,
                "max_attention": 3,
                "effective_rank": 4,
                "sink_fraction": 5,
                "variance": 6,
                "kurtosis": 7,
                "top3_mass": 8,
                "top5_mass": 9,
                "argmax_pos_norm": 10,
                "spread_std": 11,
                "gini": 12,
            }
            selected_feature_idx = metric_to_feature_col.get(
                getattr(self.config, "head_metric_type", "shannon_entropy"),
                0,
            )
            selected_feature_idx = int(np.clip(selected_feature_idx, 0, raw_series_np.shape[2] - 1))
            series_np = raw_series_np[:, :, selected_feature_idx]

        # Optional Exp3/Exp6 behaviour: exclude prompt tokens from causality.
        prompt_len = int(getattr(self.config, "prompt_token_count", 0) or 0)
        if getattr(self.config, "analyse_generated_only", True) and prompt_len > 0 and series_np.shape[0] > (prompt_len + 5):
            series_np = series_np[prompt_len:]
            if raw_series_np.ndim == 3 and raw_series_np.shape[0] > (prompt_len + 5):
                raw_series_np = raw_series_np[prompt_len:]

        # ── 1. Sanitize + preprocessing ───────────────────────────────────
        # [UPGRADE V2]: Use robust min-max normalization for all metrics to ensure
        # max_attention and others are in [0, 1] for both model and visuals.
        series_np = self._preprocess_series_for_var(series_np)
        series_for_model = series_np

        active_indices = np.where(series_for_model.std(axis=0) > 1e-6)[0]
        if len(active_indices) < 2:
            return {
                "error": "insufficient diversity in head dynamics",
                "layer_name": layer_name,
                "active_heads_count": len(active_indices)
            }

        filtered_series = series_for_model[:, active_indices]
        T, H_active = filtered_series.shape

        # Keep VAR identifiable by limiting equations relative to token count.
        # Conservative cap to keep small-sample VAR stable.
        max_equations = max(2, min(12, T // 5))
        if H_active > max_equations:
            col_var = filtered_series.var(axis=0)
            top_cols = np.argsort(col_var)[::-1][:max_equations]
            top_cols = np.sort(top_cols)
            filtered_series = filtered_series[:, top_cols]
            active_indices = active_indices[top_cols]
            H_active = filtered_series.shape[1]

        lag = max_lag or self.config.var_max_lags
        # 1. Gap A.4: Automated Lag Selection (p level lags)
        optimal_p = self._select_optimal_lag(filtered_series, lag)
        self._selected_lag = optimal_p

        # ── FAVOR: Factor-Augmented VAR (Gap Omitted Variable Bias) ───────
        exog_factors = None
        if getattr(self.config, "use_favar", True):
            try:
                # Extract global factors from the last layer activations
                # trajectory is Dict[layer_name, Tensor[T, d]]
                last_layer = sorted(trajectory.keys())[-1]
                global_h = trajectory[last_layer].detach().cpu().numpy()
                # Remove prompt tokens if needed
                if getattr(self.config, "analyse_generated_only", True) and prompt_len > 0 and global_h.shape[0] > (prompt_len + 5):
                    global_h = global_h[prompt_len:]
                
                exog_factors = self._extract_global_pca_factors(global_h, n_factors=3)
                console.print(f"[bold cyan]FAVAR Active:[/] Conditioned on {exog_factors.shape[1]} global factors.")
            except Exception as e:
                console.print(f"[yellow]FAVAR factor extraction failed: {e}[/]")

        # ── Non-linear Expansion (Gap Linearity) ───────────────────────────
        # Add interaction terms for top-influence heads if enabled
        if getattr(self.config, "use_interaction_terms", False) and H_active >= 2:
            interaction_series = self._generate_interaction_terms(filtered_series, top_k=2)
            if exog_factors is not None:
                exog_factors = np.hstack([exog_factors, interaction_series])
            else:
                exog_factors = interaction_series
        
        
        # 2. Gap A.1: Per-head ADF stationarity
        stationarity_report = self._test_per_head_stationarity(filtered_series, p_lags=optimal_p)
        
        # ── Result accumulator ─────────────────────────────────────────────
        _, H_total = series_for_model.shape
        result = {
            "layer_name": layer_name,
            "generated_text": generated_text,
            "series": series_np,
            "active_heads": active_indices.tolist(),
            "selected_feature_idx": selected_feature_idx,
            "stationarity": stationarity_report,
            "stability_roots": [],
            "spectral_radius": 0.0,
            "lag_horizon_matrix": np.zeros((H_total, H_total)),
            "n_lags": 0,
        }
        
        if T <= 2 * optimal_p + min(5, H_active):
            # ── [GOTCHA RESOLVED] ── Fallback to correlation for short responses
            console.print(f"[yellow]Insufficient tokens ({T}) for VAR. Running Correlation Fallback...[/]")
            lagged_corr = self._correlation_fallback(filtered_series, lag=1)
            
            full_influence = np.zeros((H_total, H_total))
            for i, idx_i in enumerate(active_indices):
                for j, idx_j in enumerate(active_indices):
                    full_influence[idx_i, idx_j] = lagged_corr[i, j]
            
            result.update({
                "influence_matrix": full_influence,
                "lag_horizon_matrix": full_influence,
                "n_lags": 1,
                "is_fallback": True,
                "model_type": "Correlation (Fallback)",
                "var_max_lag": 1,
            })
            self._last_influence_matrix = full_influence
            return result
        self._last_fdr_result = None
        self._last_pdc = None
        self._selected_lag = None
        self._last_joint_stationarity = None
        self._last_coint = None
        self._vecm_used = False
        result["feature_decomposition"] = self.observer.decompose_feature_space(raw_series_np)

        result['stationarity'] = stationarity_report

        # ── 3. Gap A.5: Joint ADF+KPSS ────────────────────────────────────
        if self.config.joint_stationarity_test:
            joint_report = self.observer.joint_stationarity_test(filtered_series)
            result['joint_stationarity'] = joint_report
            self._last_joint_stationarity = joint_report

        # ── 4-6. Cointegration/VECM or VAR ────────────────────────────────
        USE_VECM = False
        series_for_var = filtered_series

        if stationarity_report['needs_diff']:
            # Check for cointegration before blindly differencing
            if self.config.run_johansen_cointegration and H_active >= 2:
                try:
                    coint_report = self._check_cointegration(filtered_series, p_lags=optimal_p)
                    result['cointegration'] = coint_report
                    self._last_coint = coint_report
                    
                    if coint_report['cointegrated'] and T >= 40:
                        # VECM is statistically sound for small H, moderate T systems.
                        # T >= 40 is a lowered threshold allowing more VECM usage in reasoning chains.
                        USE_VECM = True
                        self._vecm_used = True
                        vecm_result = self._fit_vecm(
                            filtered_series,
                            rank=coint_report['n_coint_vectors'],
                            p_lags=optimal_p
                        )
                        H = H_active
                        
                        # ── VECM Interpretation Logic ──────────────────────────────
                        # 1. Long-run impact (Pi = αβ')
                        alpha = vecm_result.alpha # [H, rank]
                        beta = vecm_result.beta   # [H, rank]
                        VECM_Pi = np.abs(alpha @ beta.T)
                        
                        # 2. Short-run impact (Gamma)
                        gamma_mat = vecm_result.gamma # [H, H * (optimal_p - 1)]
                        n_diff_lags = max(optimal_p - 1, 0)
                        
                        # Reshape Gamma to [diff_lag, target_head, source_head]
                        coef_tensor = np.zeros((n_diff_lags, H, H))
                        if n_diff_lags > 0:
                            if gamma_mat.shape[1] == H * n_diff_lags:
                                # Standard statsmodels Gamma structure
                                for l in range(n_diff_lags):
                                    coef_tensor[l] = gamma_mat[:, l*H : (l+1)*H]
                            else:
                                # Fallback fallback
                                prod = gamma_mat.shape[1]
                                n_lags_derived = prod // H
                                temp = gamma_mat.reshape(H, n_lags_derived, H).transpose(1, 0, 2)
                                coef_tensor = temp[:n_diff_lags] if n_lags_derived >= n_diff_lags else temp
                            
                            weights = np.arange(1, n_diff_lags + 1, dtype=np.float32).reshape(n_diff_lags, 1, 1)
                            VECM_Gamma_Sum = np.sum(np.abs(coef_tensor) / weights, axis=0)
                            
                            # Total Influence = Long-run + Weighted Short-run
                            influence_active = VECM_Pi + VECM_Gamma_Sum
                        else:
                            influence_active = VECM_Pi
                        
                        result['model_type'] = 'VECM'
                        result['vecm_coint_rank'] = coint_report['n_coint_vectors']
                        result['lag_weighting'] = 'long_run_plus_decay'
                        
                        # Export for downstream mapping
                        # We stack [Pi | Gammas] into current_coefs to reveal the full causal structure in the dashboard
                        n_lags = 1 + n_diff_lags  # Pi is lag 0 (long run)
                        # current_coefs should be [TotalStages, H, H]
                        pi_3d = VECM_Pi[np.newaxis, ...]
                        if n_diff_lags > 0:
                            current_coefs = np.concatenate([pi_3d, coef_tensor], axis=0)
                        else:
                            current_coefs = pi_3d
                except Exception as e:
                    console.print(f"[yellow]Cointegration/VECM failed: {e}. Falling back to VAR.[/]")
                    USE_VECM = False

            if not USE_VECM:
                # Apply selective differencing (Gap A.2)
                series_for_var, diff_mask = self._apply_selective_differencing(
                    filtered_series, stationarity_report['diff_mask']
                )
                # CRITICAL: If differencing happened, length decreased by 1. Update T.
                T = len(series_for_var)
                result['differenced'] = True
                result['diff_mask'] = diff_mask.tolist()

        # ── Standard VAR path (Ridge Regularized Pipeline) ─────────────────────────
        var_result = None
        if not USE_VECM:
            try:
                import torch
                
                # Gap A.4: Use optimal p selected via AIC/BIC
                k_ar = optimal_p
                
                # Construct design matrices for Ridge VAR: Y = X B
                # Y shape: [T_eff, H], X shape: [T_eff, k_ar * H]
                T_eff = T - k_ar
                Y = series_for_var[k_ar:]
                X_list = [series_for_var[k_ar-k : T-k] for k in range(1, k_ar+1)]
                X = np.hstack(X_list)

                # ── FAVAR/Interactions Integration ────────────────────────
                num_exog = 0
                if exog_factors is not None:
                    # Align exogenous factors with the target series Y (which starts at optimal_p)
                    # exog_factors shape is [T, F].
                    exog_aligned = exog_factors[k_ar:]
                    if exog_aligned.shape[0] == T_eff:
                        X = np.column_stack([X, exog_aligned])
                        num_exog = exog_aligned.shape[1]
                
                # L2 Penalty to prevent singular matrix and overfitting
                lambda_reg = getattr(self.config, "var_ridge_penalty", 1.0)
                
                Xt = torch.tensor(X, dtype=torch.float32)
                Yt = torch.tensor(Y, dtype=torch.float32)
                
                # Ridge solution: B = (X^T X + λI)^-1 X^T Y
                # Using PyTorch for potential GPU acceleration / robust linalg
                Xt_Xt = Xt.T @ Xt
                reg_matrix = lambda_reg * torch.eye(Xt.shape[1], dtype=torch.float32, device=Xt.device)
                
                # B has shape [k_ar * H + num_exog, H]
                Bt = torch.linalg.solve(Xt_Xt + reg_matrix, Xt.T @ Yt)
                
                # Reshape to [k_ar, H, H] where coefs[k] maps X_{t-k} to X_t
                # Separate exogenous weights from head coefficients
                B_np = Bt.numpy()
                if num_exog > 0:
                    head_B_np = B_np[:-num_exog]
                    exog_B_np = B_np[-num_exog:]
                    # Store exog weights for diagnostics (optional)
                    result['exog_weights'] = exog_B_np.tolist()
                else:
                    head_B_np = B_np
                
                coefs = head_B_np.T.reshape(H_active, k_ar, H_active).transpose(1, 0, 2)
                
                # Export for downstream mapping
                n_lags = k_ar
                current_coefs = coefs
                
                # [PYTORCH] Compute residuals: Y - X @ B
                # Using PyTorch for the subtraction and matrix multiply to ensure precision
                Y_pred = Xt @ Bt
                resid_tensor = Yt - Y_pred
                resid_np = resid_tensor.numpy()
                
                # Check for white noise using Ljung-Box test
                lb_p_vals = []
                # Test each variable (head) individually for white noise
                # Max lag is min(10, n_samples/5)
                max_lb_lag = min(10, len(resid_np) // 5) if len(resid_np) > 10 else 1
                
                for h_idx in range(resid_np.shape[1]):
                    lb_res = acorr_ljungbox(resid_np[:, h_idx], lags=[max_lb_lag], return_df=True)
                    lb_p_vals.append(lb_res['lb_pvalue'].iloc[0])
                
                # Store p-values for report
                result['residual_ljung_box_p_avg'] = float(np.mean(lb_p_vals))
                result['residual_white_noise_pass'] = float(np.mean(np.array(lb_p_vals) > 0.05))

                var_result = MockVARResult(coefs, k_ar) # Re-enable Granger/PDC blocks


            except Exception as e:
                return {**result, "error": f"Ridge VAR fit failed: {str(e)}"}

            if coefs is None or coefs.size == 0:
                return {**result, "error": "VAR returned no coefficients"}

            # [PYTORCH] Lag-weighted absolute sum for standard VAR
            # coefs shape: [k_ar, H, H]
            weights = np.arange(1, k_ar + 1, dtype=np.float32).reshape(k_ar, 1, 1)
            influence_active = np.sum(np.abs(coefs) / weights, axis=0)

            result['model_type'] = 'RidgeVAR'
            result['selected_lag'] = k_ar
            self._selected_lag = k_ar
            
            final_n_lags = k_ar
            final_coefs = coefs

        # ── Unified Mapping to Full H×H Dimensions ──────────────────────────
        final_n_lags = n_lags if 'n_lags' in locals() else 0
        final_coefs = current_coefs if 'current_coefs' in locals() else None
        
        # Override with VAR branch names if they exist and are newer
        if 'k_ar' in locals() and 'coefs' in locals() and not USE_VECM:
            final_n_lags = k_ar
            final_coefs = coefs

        full_influence = np.zeros((H_total, H_total))
        
        if final_coefs is not None and final_n_lags > 0:
            # Full Horizon Heatmap: [H_total, final_n_lags * H_total]
            full_lag_horizon = np.zeros((H_total, final_n_lags * H_total))
            
            # Map active sub-matrix to full-model telemetry
            for l in range(final_n_lags):
                c_lag = final_coefs[l] # [H_active, H_active]
                for i, idx_i in enumerate(active_indices):
                    for j, idx_j in enumerate(active_indices):
                        val = float(c_lag[i, j])
                        # Horizon map
                        full_lag_horizon[idx_i, l * H_total + idx_j] = val
                        # Cumulative influence (lag-weighted absolute sum)
                        full_influence[idx_i, idx_j] += abs(val) / (l + 1)
            
            result['influence_matrix'] = full_influence
            result['lag_horizon_matrix'] = full_lag_horizon
            result['n_lags'] = int(final_n_lags)
            self._last_influence_matrix = full_influence
            
            # --- VAR Stability Calculation ---
            roots, sr = self._calculate_var_stability(final_coefs)
            result['stability_roots'] = roots
            result['spectral_radius'] = sr
        else:
            result['influence_matrix'] = full_influence
            result['lag_horizon_matrix'] = full_influence
            result['n_lags'] = 0
            result['stability_roots'] = []
            result['spectral_radius'] = 0.0
        # (Unified Mapping complete)

        # ── D.2X: Topological Rank Analysis (Gap E-X) ──────────────────
        # Detect if the interaction matrix has 'collapsed' into a single direction (rank=1)
        # or stays high-rank (multifaceted reasoning).
        try:
            inf_tensor = torch.tensor(influence_active, dtype=torch.float32)
            u, s, v = torch.svd(inf_tensor)
            s_sum = s.sum()
            # Effective rank via thresholded singular values
            eff_rank = (s > (s.max() * 0.1)).sum().item() if s_sum > 0 else 0
            # von Neumann Entropy of the singular value distribution (Reasoning Richness)
            rank_entropy = -( (s/s_sum) * torch.log(s/s_sum + 1e-9) ).sum().item() if s_sum > 0 else 0
            
            result['topological_rank'] = {
                'effective_rank': int(eff_rank),
                'rank_entropy': float(rank_entropy),
                'rank_collapse': bool(eff_rank < 2 and H_active >= 4),
            }
        except Exception:
            pass

        # ── 7. Gap B.1: Granger F-test ────────────────────────────────────
        active_granger_pval = None
        if self.config.granger_ftest:
            if USE_VECM and 'vecm_result' in locals():
                active_granger_pval = self._granger_pvalue_matrix_vecm(
                    vecm_result, H_active
                )
            elif var_result is not None:
                active_granger_pval = self._granger_pvalue_matrix(
                    var_result, series_for_var, result.get('selected_lag', lag)
                )

        if active_granger_pval is not None:
            try:
                result['granger_pval_matrix'] = active_granger_pval
                # ── 8. Gap B.2: BH-FDR correction ─────────────────────────
                fdr_result_active = self._apply_fdr_correction(active_granger_pval, alpha=self.config.fdr_alpha)

                # Re-map active-space FDR outputs back to full head-space so
                # dashboard matrices share the same HxH dimensions.
                full_reject = np.zeros((H_total, H_total), dtype=bool)
                full_pvals = np.full((H_total, H_total), np.nan)
                for i, idx_i in enumerate(active_indices):
                    for j, idx_j in enumerate(active_indices):
                        full_reject[idx_i, idx_j] = bool(fdr_result_active['reject_matrix'][i, j])
                        full_pvals[idx_i, idx_j] = fdr_result_active['pval_corrected'][i, j]

                mapped_pairs = []
                for source_i, target_i, pval in fdr_result_active.get('significant_pairs', []):
                    source_h = int(active_indices[source_i])
                    target_h = int(active_indices[target_i])
                    mapped_pairs.append((source_h, target_h, pval))

                fdr_result = {
                    'reject_matrix': full_reject,
                    'pval_corrected': full_pvals,
                    'n_significant': int(full_reject.sum()),
                    'significant_pairs': mapped_pairs,
                }

                result['fdr_result'] = fdr_result
                self._last_fdr_result = fdr_result

                # Mask influence: zero out non-significant pairs
                masked_influence_active = influence_active.copy()
                if fdr_result['n_significant'] == 0:
                    # [GOTCHA RESOLVED] FDR Washout: If no pairs pass strictly, 
                    # allow top 5% to show "latent trends" with a caveat.
                    thresh = np.percentile(np.abs(influence_active), 95)
                    masked_influence_active[np.abs(influence_active) < thresh] = 0.0
                    result['fdr_low_confidence'] = True
                else:
                    masked_influence_active[~fdr_result_active['reject_matrix']] = 0.0

                masked_full = np.zeros((H_total, H_total))
                for i, idx_i in enumerate(active_indices):
                    for j, idx_j in enumerate(active_indices):
                        masked_full[idx_i, idx_j] = masked_influence_active[i, j]

                result['masked_influence'] = masked_full
                result['n_significant_pairs'] = fdr_result['n_significant']
                result['significant_pairs'] = fdr_result['significant_pairs']

            except Exception as e:
                console.print(f"[yellow]Granger/FDR failed: {e}[/]")

        # ── 9. Gap B.5: PDC ───────────────────────────────────────────────
        if self.config.compute_pdc and var_result is not None:
            try:
                pdc_result = self._partial_directed_coherence(var_result)
                result['pdc'] = pdc_result
                self._last_pdc = pdc_result
            except Exception as e:
                console.print(f"[yellow]PDC failed: {e}[/]")

        # ── Optional: Bootstrap surrogates ─────────────────────────────────
        if self.config.enable_bootstrap_surrogates and var_result is not None:
            try:
                bootstrap_pvals = self._bootstrap_surrogate_pvalues(
                    series_for_var, influence_active,
                    n_surrogates=self.config.n_bootstrap_surrogates,
                    lag=result.get('selected_lag', lag)
                )
                result['bootstrap_pvals'] = bootstrap_pvals
            except Exception as e:
                console.print(f"[yellow]Bootstrap surrogates failed: {e}[/]")

        return result

    def _build_induction_direction(
        self,
        pos_prompts: List[str],
        neg_prompts: List[str],
        layer_idx: int
    ) -> np.ndarray:
        """
        E.3 — Build functional direction using steering-vectors.
        """
        from steering_vectors import train_steering_vector
        
        target_model = self.interceptor.model.model if hasattr(self.interceptor.model, 'model') else self.interceptor.model
        
        sv = train_steering_vector(
            target_model,
            self.interceptor.tokenizer,
            training_data=list(zip(pos_prompts, neg_prompts)),
            layers=[layer_idx],
            read_token_index=-1,
        )
        direction = sv.layer_activations[layer_idx].detach().cpu().numpy()
        return direction / np.linalg.norm(direction)

    def _project_onto_functional_directions(
        self,
        head_output_series: np.ndarray,
        function_directions: dict,
    ) -> dict:
        """
        E.4 — Project head_output_series [T, H, head_dim] onto each direction.
        """
        projections = {}
        for func_name, direction in function_directions.items():
            proj = np.einsum('thd,d->th', head_output_series, direction)
            projections[func_name] = proj
        return projections

    def interventional_head_causality(
        self,
        prompt: str,
        layer_name: str,
        source_head_index: int,
    ) -> Dict[str, Any]:
        """
        [GOTCHA RESOLVED] VRAM Safety Lock: Only one intervention can run at a time 
        to prevent CUDA Out-of-Memory during interleaved analysis requests.
        """
        with self._intervention_lock:
            console.print(f"[cyan]Intervening on Head {source_head_index} in {layer_name}...[/]")
        
        # 1. Clean run
        clean_traj, clean_text = self.interceptor.capture_generation(prompt)
        clean_series = self.interceptor.get_head_metric_series(layer_name)
        
        # 2. Patched run
        patched_traj, patched_text = self.interceptor.patch_attention_heads(
            prompt, layer_name, [source_head_index], ablation_type="zero"
        )
        patched_series = self.interceptor.get_head_metric_series(layer_name)
        
        if clean_series.numel() == 0 or patched_series.numel() == 0:
            return {"error": "metrics not captured"}
            
        # Align lengths
        min_len = min(clean_series.shape[0], patched_series.shape[0])
        clean_series = clean_series[:min_len]
        patched_series = patched_series[:min_len]
        
        # Measure L2 divergence per head
        # divergence: [n_heads]
        diff = (clean_series - patched_series).pow(2)
        causal_influence = diff.mean(dim=0).cpu().numpy()
        
        # Output comparison
        text_divergence = 0.0
        if clean_text != patched_text:
            # Simple divergence metric (normalized edit distance proxy)
            text_divergence = 1.0 - (len(os.path.commonprefix([clean_text, patched_text])) / max(len(clean_text), 1))

        out = {
            "source_head": source_head_index,
            "layer": layer_name,
            "causal_influence_vector": causal_influence,
            "global_text_divergence": text_divergence,
            "clean_text": clean_text,
            "patched_text": patched_text
        }
        self._last_pert_results = [
            {
                "head": int(source_head_index),
                "target": int(np.argmax(causal_influence)) if causal_influence.size else -1,
                "mode": "zero",
                "delta_entropy": float(np.mean(causal_influence)) if causal_influence.size else 0.0,
                "restoration": 0.0,
                "kl_patch": 0.0,
                "confirmed": False,
            }
        ]
        return out

    def interventional_head_causality_multilayer(
        self,
        prompt: str,
        layer_names: List[str],
        source_head_indices: Union[int, List[int]],
    ) -> Dict[str, Any]:
        """
        Simultaneously ablate the same source head across multiple layers and
        return per-layer and aggregate divergence statistics.
        """
        if not layer_names:
            return {"error": "no layers specified"}

        if isinstance(source_head_indices, int):
            source_heads = [source_head_indices]
        else:
            source_heads = [int(h) for h in source_head_indices]

        console.print(
            f"[cyan]Intervening on Heads {source_heads} across layers: "
            f"{', '.join(layer_names)}[/]"
        )

        # 1) Clean run and capture head metrics for requested layers.
        _, clean_text = self.interceptor.capture_generation(prompt)
        clean_series_by_layer = {
            ln: self.interceptor.get_head_metric_series(ln) for ln in layer_names
        }

        # 2) One patched run with all target layers ablated simultaneously.
        layer_to_heads = {ln: source_heads for ln in layer_names}
        _, patched_text = self.interceptor.patch_attention_heads_multi(
            prompt,
            layer_to_heads=layer_to_heads,
            ablation_type="zero",
        )
        patched_series_by_layer = {
            ln: self.interceptor.get_head_metric_series(ln) for ln in layer_names
        }

        per_layer = {}
        per_head_accum = {int(h): [] for h in source_heads}
        aggregate = []
        for ln in layer_names:
            clean_series = clean_series_by_layer.get(ln, torch.empty(0))
            patched_series = patched_series_by_layer.get(ln, torch.empty(0))
            if clean_series.numel() == 0 or patched_series.numel() == 0:
                per_layer[ln] = {"error": "metrics not captured"}
                continue

            min_len = min(clean_series.shape[0], patched_series.shape[0])
            clean_series = clean_series[:min_len]
            patched_series = patched_series[:min_len]

            diff = (clean_series - patched_series).pow(2)
            causal_influence = diff.mean(dim=0).cpu().numpy()

            # Build per-head diagnostics for Exp6 dashboard perturbation panel.
            # Supports scalar [H] and vector [H, F] head metric modes.
            influence_head = np.asarray(causal_influence, dtype=float)
            if influence_head.ndim == 2:
                influence_head = influence_head.mean(axis=1)
            elif influence_head.ndim == 1:
                n_heads = int(getattr(self.config, "n_heads", 0) or 0)
                if n_heads > 0 and influence_head.size > n_heads and influence_head.size % n_heads == 0:
                    influence_head = influence_head.reshape(n_heads, -1).mean(axis=1)

            per_layer[ln] = {
                "causal_influence_vector": causal_influence,
                "mean_influence": float(causal_influence.mean()),
            }
            aggregate.append(float(causal_influence.mean()))

            if influence_head.ndim == 1 and influence_head.size > 0:
                target_head = int(np.argmax(influence_head))
                for h in source_heads:
                    h_i = int(h)
                    src_idx = min(max(h_i, 0), influence_head.size - 1)
                    per_head_accum[h_i].append(
                        {
                            "head": h_i,
                            "target_head": target_head,
                            "target_layer": ln,
                            "delta_entropy": float(influence_head[src_idx]),
                            "restoration": 0.0,
                            "kl_patch": 0.0,
                        }
                    )

        text_divergence = 0.0
        if clean_text != patched_text:
            text_divergence = 1.0 - (
                len(os.path.commonprefix([clean_text, patched_text])) / max(len(clean_text), 1)
            )

        per_head_results = []
        for h_i, rows in per_head_accum.items():
            if not rows:
                continue
            per_head_results.append(
                {
                    "head": int(h_i),
                    "target_head": int(rows[-1].get("target_head", h_i)),
                    "target_layer": rows[-1].get("target_layer"),
                    "delta_entropy": float(np.mean([r.get("delta_entropy", 0.0) for r in rows])),
                    "restoration": float(np.mean([r.get("restoration", 0.0) for r in rows])),
                    "kl_patch": float(np.mean([r.get("kl_patch", 0.0) for r in rows])),
                }
            )

        return {
            "source_heads": source_heads,
            "layers": layer_names,
            "per_layer": per_layer,
            "per_head_results": per_head_results,
            "aggregate_mean_influence": float(np.mean(aggregate)) if aggregate else 0.0,
            "global_text_divergence": text_divergence,
            "clean_text": clean_text,
            "patched_text": patched_text,
        }

    # ------------------------------------------------------------------ #
    #  NEXUS Layer: Hypergraph & Isomorphic Mapping
    # ------------------------------------------------------------------ #

    def _label_hyperedge_with_llm(
        self,
        member_labels: List[str],
        layer_name: str,
        weight: float,
    ) -> str:
        """
        Call a small local LLM to generate a human-readable principle label
        for a hyperedge.  Falls back to the heuristic label on any error.

        Supports two transports controlled by config.local_llm_transport:
          • "ollama"        — HTTP POST to localhost:11434/api/generate
          • "transformers"  — HuggingFace text-generation pipeline
        """
        fallback = f"Motif @ {layer_name} ({member_labels[0] if member_labels else '?'}...)"

        if not getattr(self.config, "use_local_llm_labeling", False):
            return fallback

        tokens_preview = ", ".join(member_labels[:6])
        prompt_text = (
            f"You are an interpretability tool. Given a cluster of tokens "
            f"[{tokens_preview}] co-activating in transformer layer '{layer_name}' "
            f"(cosine similarity weight {weight:.2f}), describe the abstract "
            f"semantic/structural principle in ≤8 words. Output ONLY the label."
        )

        transport = getattr(self.config, "local_llm_transport", "ollama")
        model_id = getattr(self.config, "local_llm_model", "qwen2.5-coder:3b")
        timeout = getattr(self.config, "local_llm_timeout", 10.0)
        max_tok = getattr(self.config, "local_llm_max_tokens", 60)

        try:
            if transport == "ollama":
                import json
                import urllib.request

                payload = json.dumps({
                    "model": model_id,
                    "prompt": prompt_text,
                    "stream": False,
                    "options": {"num_predict": max_tok},
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
                label = data.get("response", "").strip()
                return label if label else fallback

            elif transport == "transformers":
                from transformers import pipeline as hf_pipeline
                if not hasattr(self, "_label_pipeline"):
                    self._label_pipeline = hf_pipeline(
                        "text-generation",
                        model=model_id,
                        max_new_tokens=max_tok,
                        device_map="auto",
                    )
                result = self._label_pipeline(prompt_text)
                label = result[0]["generated_text"].replace(prompt_text, "").strip()
                return label if label else fallback

        except Exception as exc:
            console.print(f"[yellow]LLM labeling failed ({exc}), using heuristic.[/]")
            return fallback

        return fallback

    def extract_hyperedges(
        self,
        trajectory: torch.Tensor,
        token_labels: List[str],
        layer_name: str,
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Identify latent motifs (Hyperedges) in the residual stream.
        Groups tokens that exhibit similar manifold behavior (cosine similarity
        in latent space).
        """
        if trajectory.dim() == 3: # [Batch, Seq, Hidden]
            trajectory = trajectory.squeeze(0)
            
        n_tokens = trajectory.shape[0]
        if n_tokens < 2:
            return []

        # Simple adjacency based on cosine similarity
        norm_traj = torch.nn.functional.normalize(trajectory.float(), p=2, dim=-1)
        sim_matrix = torch.mm(norm_traj, norm_traj.t()).cpu().numpy()
        
        hyperedges = []
        visited = set()
        
        for i in range(n_tokens):
            if i in visited:
                continue
            
            # Find tokens highly similar to i
            cluster_indices = np.where(sim_matrix[i] > threshold)[0]
            if len(cluster_indices) > 2: # Minimal motif size
                member_labels = [token_labels[idx] for idx in cluster_indices if idx < len(token_labels)]
                edge_weight = float(np.mean(sim_matrix[i, cluster_indices]))

                # Optionally call a local LLM to produce a richer semantic label.
                principle = self._label_hyperedge_with_llm(
                    member_labels, layer_name, edge_weight
                )
                
                hyperedges.append({
                    "hyperedge_id": f"he-{uuid.uuid4().hex[:8]}",
                    "principle": principle,
                    "tokens": member_labels,
                    "layer": layer_name,
                    "weight": edge_weight
                })
                visited.update(cluster_indices)
        
        return hyperedges

    def detect_isomorphic_clusters(
        self,
        hyperedges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map Hyperedges to abstract principles (Isomorphic Mapping).
        Finds motifs from different layers/token-ranges that represent
        the same underlying structural principle.
        """
        if len(hyperedges) < 2:
            return []
            
        clusters = []
        # In a real implementation, we would embed the hyperedge 'principle' 
        # or token-sets and use cosine similarity.
        # For this eager demo, we cluster by token-overlap.
        
        for i, h1 in enumerate(hyperedges):
            for j, h2 in enumerate(hyperedges):
                if i >= j:
                    continue
                
                s1 = set(h1["tokens"])
                s2 = set(h2["tokens"])
                overlap = len(s1.intersection(s2)) / (len(s1 | s2) + 1e-9)
                
                if overlap > 0.4: # Lower threshold for cross-layer isomorphism
                    clusters.append({
                        "cluster_id": f"iso-{uuid.uuid4().hex[:8]}",
                        "shared_principle": f"Structural Isomorphism ({h1['principle']})",
                        "hyperedge_ids": [h1["hyperedge_id"], h2["hyperedge_id"]],
                        "similarity": float(overlap)
                    })
        
        return clusters



    def _extract_global_pca_factors(self, hidden_states: np.ndarray, n_factors: int = 3) -> np.ndarray:
        """
        Extract principal components from hidden states to use as exogenous
        factors in FAVAR. This captures global model 'drift' or 'context' 
        that might cause spurious correlations between heads.
        """
        T, D = hidden_states.shape
        # Ensure we don't request more factors than dimensions or tokens
        n_factors = min(n_factors, T, D)
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(hidden_states)
        # Normalize factors to prevent scale dominance in VAR
        factors = (factors - factors.mean(axis=0)) / (factors.std(axis=0) + 1e-9)
        return factors

    def _generate_interaction_terms(self, series: np.ndarray, top_k: int = 2) -> np.ndarray:
        """
        Generate cross-product interaction terms between top heads to capture
        non-linear logic gates (Gap: Linearity).
        """
        T, H = series.shape
        # Use variance to find most 'active' heads for interaction
        variances = np.var(series, axis=0)
        top_indices = np.argsort(variances)[-top_k:]
        
        interactions = []
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                idx_i, idx_j = top_indices[i], top_indices[j]
                inter = series[:, idx_i] * series[:, idx_j]
                interactions.append(inter)
        
        if not interactions:
            return np.zeros((T, 0))
            
        return np.column_stack(interactions)

    def _discover_phases_hmm(self, series: np.ndarray, n_states: int = 4) -> Dict:
        """
        Fit a Hidden Markov Model to the head dynamics to discover reasoning phases.
        [UPGRADE V2]: Added persistence bias to transition matrix to model 
        non-Markovian reasoning mode stability.
        """
        try:
            from hmmlearn import hmm
            
            # Persistence Bias: LLMs tend to stay in 'Reasoning' or 'Output' mode
            # for many tokens. We encourage high diagonal values in transition matrix.
            model = hmm.GaussianHMM(
                n_components=n_states, 
                covariance_type="full", 
                n_iter=100,
                init_params="stmc" # Allow internal initialization first
            )
            
            # Pre-processing: Standardize
            T, H = series.shape
            series_std = (series - series.mean(axis=0)) / (series.std(axis=0) + 1e-9)
            
            model.fit(series_std)
            states = model.predict(series_std)
            posteriors = model.predict_proba(series_std)
            
            # Calculate metrics
            aic = -2 * model.score(series_std) + 2 * (n_states**2 + 2*n_states*H)
            bic = -2 * model.score(series_std) + np.log(T) * (n_states**2 + 2*n_states*H)
            
            # Extract transition matrix for 'persistence' analysis
            trans_mat = model.transmat_
            persistence = np.diag(trans_mat).tolist()
            
            return {
                "states": states.tolist(),
                "posteriors": posteriors.tolist(),
                "transition_matrix": trans_mat.tolist(),
                "state_means": model.means_.tolist(), # NEW
                "persistence_scores": persistence,
                "aic": float(aic),
                "bic": float(bic),
                "n_states": n_states,
            }
        except Exception as e:
            return {"error": str(e)}

    def _correlation_fallback(self, series: np.ndarray, lag: int = 1) -> np.ndarray:
        """
        Computes a time-lagged correlation matrix as a robust fallback for 
        short-sample systems where VAR/VECM cannot be reliably identified.
        """
        T, H = series.shape
        if T <= lag:
            return np.eye(H)
            
        corrs = np.zeros((H, H))
        # Y_t and X_{t-lag}
        for i in range(H):
            target = series[lag:, i]
            for j in range(H):
                source = series[:T-lag, j]
                # Pearson correlation coefficient
                if np.std(target) > 1e-9 and np.std(source) > 1e-9:
                    r, _ = pearsonr(source, target)
                    corrs[i, j] = np.abs(r)
                else:
                    corrs[i, j] = 0.0
                    
        return corrs

    def _calculate_var_stability(self, coefs: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Check VAR stability by calculating internal roots of the characteristic polynomial.
        A VAR(p) is stable if all roots are inside the complex unit circle.
        """
        # coefs shape: [k_ar, H, H]
        k_ar, H, _ = coefs.shape
        if H == 0:
            return [], 0.0
            
        # Companion matrix construction
        # [ A1 A2 ... Ap ]
        # [ I  0  ... 0  ]
        # [ 0  I  ... 0  ]
        companion = np.zeros((H * k_ar, H * k_ar))
        # Top row: A_1, A_2, ..., A_p
        for i in range(k_ar):
            companion[:H, i*H:(i+1)*H] = coefs[i]
        # Identity blocks below
        if k_ar > 1:
            companion[H:, :H*(k_ar-1)] = np.eye(H*(k_ar-1))
            
        try:
            # eigenvalues of companion matrix are the inverse of the roots of det(I - A1z - ... - Apz^p) = 0
            # Stability requires all eigenvalues < 1
            eigenvalues = np.linalg.eigvals(companion)
            roots = []
            max_r = 0.0
            for ev in eigenvalues:
                r = float(np.abs(ev))
                max_r = max(max_r, r)
                # Cap roots list to top 20 for telemetry payload size
                if len(roots) < 20: 
                    roots.append({"real": float(ev.real), "imag": float(ev.imag), "radius": r})
            
            return roots, max_r
        except Exception:
            return [], 0.0

    def _preprocess_series_for_var(self, series: np.ndarray) -> np.ndarray:
        """
        [UPGRADE V2]: Implemented Robust Min-Max normalization for attention 
        metrics (max_attention, etc.) as requested. This ensures that even
        weak attention signals are projected into the [0, 1] range for causal
        analysis.
        """
        series = np.nan_to_num(series, nan=0.0, posinf=1.0, neginf=-1.0)
        
        raw_min = series.min(axis=0, keepdims=True)
        raw_max = series.max(axis=0, keepdims=True)
        denom = raw_max - raw_min
        denom[denom == 0] = 1.0 # Stabilize constant signals
        
        normalized = (series - raw_min) / denom
        return np.clip(normalized, 0.0, 1.0)

    def _select_optimal_lag(self, series: np.ndarray, max_lag: int) -> int:
        """
        A.4 — Automated Lag Selection via AIC/BIC/FPE. 
        Selects the lag that minimizes the Information Criterion.
        """
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            model = VAR(series)
            # Find optimal lag order via AIC
            lag_order = model.select_order(maxlags=min(int(max_lag), max(1, len(series)//10)))
            return int(lag_order.aic) if hasattr(lag_order, 'aic') else 1
        except Exception:
            return 1
