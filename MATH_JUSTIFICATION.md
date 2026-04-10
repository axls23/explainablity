# Chronoscope: Code-Grounded Justification of Mathematical Tools

**Subtitle:** Why each analytical method was selected for modeling LLM internal activities as a time series, with explicit references to implemented code components.

---

## 1. The Core Problem: LLM Activations as a Time Series

When a transformer generates a response token by token, the hidden state at each layer evolves with each new token. In Chronoscope, the `ChronoscopeInterceptor` (`chronoscope/interceptor.py`) attaches PyTorch forward hooks to every decoder layer during generation:

```python
# chronoscope/interceptor.py — _make_hook()
def hook_fn(module, input, output):
    hidden = output[0]  # [Batch, SeqLen, HiddenDim]
    # Detach + CPU to avoid GPU memory blowup during long traces
```

By capturing the last-token hidden state at each generation step, the interceptor accumulates a matrix **H ∈ ℝ^{T × D}**, where T is the number of generated tokens and D is the model's hidden dimension (e.g., D = 896 for Qwen2.5-0.5B). This matrix is **exactly a multivariate time series**: dimension is the "channel" and token-position is "time."

The problem statement — *model LLM internal activities as a time series* — therefore maps directly to analysing **H** using classical signal-processing tools. The sections below justify each tool with a specific reference to the code that implements it.

---

## 2. Pipeline Overview

```
Token t arrives
      │
      ▼
ChronoscopeInterceptor._make_hook()     ← interceptor.py
  Captures h[t] ∈ ℝ^D at each layer
      │
      ▼
Trajectory H ∈ ℝ^{T × D}              ← accumulated tensor
      │
      ▼
SignalObserver.svd_compress()           ← observer.py
  H̃ ∈ ℝ^{T × k}  (k=8 default)
      │
  ┌───┴───────────────────────────┐
  │   SignalObserver methods      │
  │   spectral_analysis()         │
  │   autocorrelation_analysis()  │
  │   stationarity_test()         │
  │   decompose()                 │
  │   calculate_trajectory_       │
  │     dynamics()                │
  │   compute_intrinsic_time()    │
  └───┬───────────────────────────┘
      │
      ▼
CausalAnalyzer methods                  ← analyzer.py
  _test_per_head_stationarity()
  _check_cointegration()
  head_interaction_analysis() → VAR / VECM
  _granger_pvalue_matrix()
  _apply_fdr_correction()
  _partial_directed_coherence()
  dtw_divergence()
  topological_analysis()
  compute_fidelity_score()
```

---

## 3. Mathematical Tool Justifications

### 3.1 SVD Compression (Dimensionality Reduction)

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/observer.py` → `SignalObserver.svd_compress()` |
| **Called by** | `SignalObserver.full_analysis()` (Step 1); also directly by `CausalAnalyzer.sensitivity_analysis_sweep()` |
| **Config** | `ChronoscopeConfig.svd_components = 8` |

**What it does:**  
Factorises the [T × D] trajectory matrix via economy SVD (**X = UΣVᵀ**) and projects onto the top *k* principal components, yielding **H̃ ∈ ℝ^{T × k}**. A Hankel-like delay embedding is optionally applied first (`svd_delay_embedding`), building a richer phase-space representation.

```python
# observer.py — svd_compress()
U, S, Vt = np.linalg.svd(X_embedded, full_matrices=False)
compressed[:, :k] = U[:, :k] * S[:k]          # [T, k] principal scores
```

**Why it is appropriate:**  
The hidden dimension D can be 896+ for even a small model. Raw time-series tools applied to 896 correlated channels would be numerically unstable, computationally intractable, and statistically meaningless (extreme multicollinearity). SVD solves all three:

1. **Tractability** — reduces D → k = 8 before any downstream analysis.  
2. **Interpretability** — each SVD axis corresponds to an orthogonal "mode of variation" in the latent space; the first axis captures the dominant *reasoning direction*.  
3. **Determinism** — unlike t-SNE or UMAP, SVD is algebraically deterministic, which means results are reproducible and comparable across runs.  
4. **Inverse mapping** — the `components` matrix (Vᵀ) maps each principal axis back to specific neuron groups, enabling mechanistic interpretability (referenced in `Chronoscope_Math_Whitepaper.md`).

---

### 3.2 Spectral Analysis — Welch's Power Spectral Density

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/observer.py` → `SignalObserver.spectral_analysis()` |
| **Called by** | `SignalObserver.full_analysis()` (Step 2); `SignalObserver.decompose()` (for period auto-detection) |
| **Config** | `ChronoscopeConfig.fft_top_k = 5` |

**What it does:**  
Applies Welch's method (overlapping-window averaged periodogram) via `scipy.signal.welch` to each of the *k* SVD components. Returns the power spectral density (PSD) and identifies the top-K dominant frequency bins.

```python
# observer.py — spectral_analysis()
from scipy.signal import welch
nperseg = min(256, max(len(signal) // 2, 8))
freqs, power = welch(signal, fs=1.0, nperseg=nperseg)
top_indices = np.argsort(power[1:])[-top_k:] + 1   # skip DC
dominant_periods = np.where(dominant_freqs > 0, 1.0 / dominant_freqs, np.inf)
```

**Why it is appropriate:**  
- **Detecting semantic periodicity** — LLM reasoning can exhibit recurring patterns (periodic revisiting of a premise, structured list generation). The spectral peaks reveal the *token-period* of these cycles. A dominant period of, say, 8 tokens indicates the model performs one conceptual sub-step every 8 generation steps.  
- **Welch over raw FFT** — raw periodograms have high variance. Welch's method averages over overlapping windows, reducing variance while preserving frequency resolution. This is critical for short token sequences (T ≈ 50–512).  
- **Spectral coherence as quality signal** — the ratio of peak power to total power (`peak_ratio` in `compute_fidelity_score()`) quantifies how *structured* the reasoning is. A high-peakedness spectrum (mostly one dominant frequency) signals structured, rhythmic reasoning. A flat spectrum signals noise or random-walk behaviour.
- **Bootstrapped phase-scrambling** (in `_bootstrap_surrogate_pvalues()`) uses `torch.fft.rfft` to verify that observed spectral peaks are statistically significant rather than artefacts of the short sequence length.

---

### 3.3 Autocorrelation — ACF and PACF

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/observer.py` → `SignalObserver.autocorrelation_analysis()` |
| **Called by** | `SignalObserver.full_analysis()` (Step 3) |
| **Config** | `ChronoscopeConfig.acf_max_lag = 20` |

**What it does:**  
Computes the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) on the first (dominant) SVD component up to `acf_max_lag` lags. Identifies *significant* lags exceeding the 95 % confidence band `2/√T`.

```python
# observer.py — autocorrelation_analysis()
from statsmodels.tsa.stattools import acf, pacf
acf_vals  = acf(signal, nlags=max_lag, fft=True)    # FFT-accelerated
pacf_vals = pacf(signal, nlags=max_lag, method="ywm")
threshold = 2.0 / np.sqrt(len(signal))
significant = np.where(np.abs(acf_vals[1:]) > threshold)[0] + 1
```

**Why it is appropriate:**  
- **Measuring attention look-back** — transformer attention mechanisms create temporal dependencies: token t attends to tokens t−k for various k. The ACF directly measures which lags k carry significant dependence in the hidden-state trajectory, providing a *signal-level mirror of the attention look-back window*.  
- **ACF vs PACF for model order** — ACF shows total correlation at each lag; PACF isolates the *direct* lag contribution after controlling for shorter lags. The PACF cutoff provides a data-driven order estimate for the downstream AR/VAR models.  
- **Rejecting the i.i.d. null** — significant ACF lags confirm that the hidden-state trajectory is NOT independent noise. This is a prerequisite justification for fitting any time-series model rather than treating each token's state as independent.

---

### 3.4 Stationarity Testing — ADF and KPSS

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/observer.py` → `SignalObserver.stationarity_test()`, `SignalObserver.joint_stationarity_test()` |
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer._test_per_head_stationarity()` |
| **Called by** | `SignalObserver.full_analysis()` (Step 4); head-interaction pipeline |

**What it does:**  
- For sequences T < 30: uses a *variance ratio test* (ratio of first-difference variance to level variance) as a robust small-sample alternative.  
- For T ≥ 30: runs the Augmented Dickey-Fuller (ADF) test (`statsmodels.tsa.stattools.adfuller`) with AIC-selected lag order.  
- `joint_stationarity_test()` runs both ADF and KPSS and produces a four-way diagnosis: *stationary / unit-root / structural-change / ambiguous*.

```python
# observer.py — stationarity_test()
diff_1 = np.diff(signal)
variance_ratio = np.var(diff_1) / (np.var(signal) + 1e-9)
# For T >= 30:
adf_stat, p_value, used_lag, nobs, critical_values, _ = adfuller(signal, autolag="AIC")

# observer.py — joint_stationarity_test()
adf_pval  = adfuller(s, autolag='AIC')[1]
kpss_stat, kpss_pval, _, _ = kpss(s, regression='c', nlags='auto')
```

**Why it is appropriate:**  
- **Gatekeeper for VAR** — VAR estimation requires covariance-stationarity. Applying VAR to a non-stationary (I(1)) series yields spurious regression. ADF acts as the gatekeeper: if it fails to reject the unit-root null, the series is differenced before VAR fitting (`_apply_selective_differencing()`).  
- **Stationarity as a reasoning indicator** — a *non-stationary* hidden-state trajectory (drift) means the model is *actively updating its belief state* (it is reasoning). A *stationary* trajectory means the hidden state has stabilised on an answer (or is stuck). This provides a qualitative interpretability signal beyond the statistical necessity.  
- **ADF + KPSS cross-validation** — ADF has low power against near-unit-root processes. Running KPSS jointly (opposite null hypothesis) resolves ambiguities and detects *structural breaks* (contradictory ADF-reject + KPSS-reject) that correspond to prompt-boundary shifts in reasoning.

---

### 3.5 Additive Signal Decomposition

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/observer.py` → `SignalObserver.decompose()`, `SignalObserver.decompose_feature_space()` |
| **Called by** | `SignalObserver.full_analysis()` (Step 5) |

**What it does:**  
Applies a moving-average-based additive decomposition **Y_t = T_t + S_t + R_t** (trend + seasonal + residual) to the first SVD component. The seasonal period is auto-detected from the dominant FFT period returned by `spectral_analysis()`.

```python
# observer.py — decompose()
kernel = np.ones(period) / period
trend  = np.convolve(signal, kernel, mode="same")   # moving average
detrended = signal - trend
for i in range(period):
    indices = np.arange(i, n, period)
    seasonal[indices] = detrended[indices].mean()
residual = signal - trend - seasonal
```

**Why it is appropriate:**  
Separating the hidden-state trajectory into components enables *interpretable attribution*:

- **Trend (T_t)**: the overall drift of the latent state toward a conclusion — increases monotonically in successful logical deduction chains.  
- **Seasonal (S_t)**: recurring structure at the detected period — maps to repetitive reasoning substeps (e.g., "check premise → derive → check again").  
- **Residual (R_t)**: the stochastic/unpredictable component — elevated residual variance correlates with uncertainty or hallucination risk.

The `decompose_feature_space()` variant additionally computes *delta energy* (first-difference L1 norm) and a *regime score* (z-scored delta energy), surfacing abrupt state transitions in the multivariate attention feature space.

---

### 3.6 Trajectory Dynamics: Velocity, Acceleration, and Hurst Exponent (DFA)

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/observer.py` → `SignalObserver.calculate_trajectory_dynamics()` |
| **Called by** | `SignalObserver.full_analysis()` (Step 6) |

**What it does:**  
Computes three geometric/statistical descriptors of the compressed trajectory:

1. **Velocity** — L2 norm of the step-wise difference `‖H̃[t] − H̃[t−1]‖₂`. High velocity = rapid semantic transition.  
2. **Momentum** — cosine similarity between consecutive displacement vectors. Measures *directional consistency* of reasoning.  
3. **Acceleration** — absolute change in velocity per step. Spikes flag *cognitive stress* (sudden shifts in reasoning pace).  
4. **Hurst Exponent (DFA)** — estimated via Detrended Fluctuation Analysis:

```python
# observer.py — calculate_trajectory_dynamics() → compute_hurst()
Y = np.cumsum(ts - np.mean(ts))          # Integrate mean-centered series
# Fit linear trend in each segment of size w
F_n = sqrt( mean_sq_residual_across_all_windows )
# Log-log regression: log F(n) ~ H * log(n)
coeffs = np.polyfit(log_n, log_F, 1)
hurst  = float(np.clip(coeffs[0], 0.0, 1.0))
```

**Why it is appropriate:**  
- **Velocity/Acceleration** are standard dynamical systems metrics (phase-space analysis). Applied to the SVD-compressed trajectory they indicate *when* and *how fast* the LLM transitions between semantic states.  
- **Hurst Exponent via DFA** is chosen over naive R/S analysis because R/S is biased for trending (non-stationary) signals — exactly what LLM reasoning trajectories are. DFA removes local polynomial trends before computing fluctuation scaling, making it robust to the very non-stationarity that characterises active reasoning. The interpretation is direct:
  - H > 0.5 → *persistent* (logical commitment: the model is following a coherent trajectory toward a conclusion).  
  - H ≈ 0.5 → *Brownian* (uncertain / random walk — high hallucination risk).  
  - H < 0.5 → *anti-persistent* (mean-reverting — the model keeps correcting itself).

---

### 3.7 Intrinsic Time — Arc Length and Curvature

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/observer.py` → `SignalObserver.compute_intrinsic_time()` |
| **Called by** | `SignalObserver.full_analysis()` (Step 7) |

**What it does:**  
Computes the *arc length* τ_t = Σ_{s=1}^{t} ‖h_s − h_{s-1}‖ along the hidden-state trajectory, then normalizes to τ ∈ [0,1]. Also computes discrete curvature κ_t = ‖second-difference‖ / ‖first-difference‖².

```python
# observer.py — compute_intrinsic_time()
diff     = h_t[1:] - h_t[:-1]                          # [T-1, D]
arc_steps = torch.linalg.norm(diff, dim=-1)             # [T-1]
tau_norm  = cumsum(arc_steps) / total_arc_length        # normalised intrinsic time

second_diff = h_t[2:] - 2*h_t[1:-1] + h_t[:-2]        # [T-2, D]
curvature   = ‖second_diff‖ / max(‖first_diff‖², 1e-8)
```

**Why it is appropriate:**  
Wall-clock token index is an *extrinsic* time. Different reasoning tasks consume different amounts of "cognitive effort" per token (a token completing a logical deduction moves the hidden state far; a punctuation token moves it little). Arc-length provides *intrinsic time*: equal increments correspond to equal amounts of hidden-state change. This enables fair comparison of trajectories across prompts of different lengths — curvature spikes at high-effort tokens (e.g., the deductive-conclusion token) are automatically highlighted without parameter tuning.

---

### 3.8 Vector Autoregression (VAR) and VECM

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer` — `_ensure_var_cls()`, `_select_optimal_lag()`, `_preprocess_series_for_var()`, `_fit_vecm()`, and the main `head_interaction_analysis()` / `run_head_var()` |
| **Config** | `ChronoscopeConfig.var_lag_selection = 'aic'`, `var_max_lags = 5`, `run_johansen_cointegration = True` |

**What it does:**  
Models the multivariate time series of per-head attention metrics (entropy, Rényi entropy, effective rank — captured by `ChronoscopeInterceptor` and stored in `_head_metrics`) as a VAR(p) system:

> ΔX_t = Π X_{t−1} + Σ_{i=1}^{p−1} Γ_i ΔX_{t−i} + ε_t  ← VECM form

Lag order *p* is selected by AIC/BIC via `_select_optimal_lag()`. If Johansen cointegration rank r > 0, a VECM is fitted instead of a differenced VAR.

```python
# analyzer.py — _select_optimal_lag()
model     = VAR(series_np)
selection = model.select_order(maxlags=actual_max)
optimal_p = selection.selected_orders.get('aic', 1)

# analyzer.py — _check_cointegration()
result = coint_johansen(series_np, det_order=0, k_ar_diff=k_diff)
rank   = int(np.sum(trace_stats > crit_95))

# analyzer.py — _fit_vecm()
model = VECM(series_np, k_ar_diff=k_diff, coint_rank=rank, deterministic='ci')
return model.fit()
```

**Why it is appropriate:**  
- **Multi-head coupling** — LLM attention heads are not independent; head j at step t influences head i at step t+1 through the residual stream. VAR is the standard model for *coupled* multivariate time series, directly capturing these cross-head dynamics.  
- **VECM for non-stationary reasoning** — reasoning paths are I(1) (trending toward a conclusion). When multiple heads share a long-run equilibrium (cointegration), VECM models both the short-run shocks (Γ_i) and the long-run error-correction path (Π = αβᵀ), where β gives the cointegrating vector (the "shared conclusion" the heads are converging toward).  
- **Uniform differencing** — the implementation in `_apply_selective_differencing()` differences ALL channels when any is non-stationary (a corrected version of a common mistake), maintaining a homogeneous I(0) system required by VAR theory.

---

### 3.9 Johansen Cointegration Test

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer._check_cointegration()` |

**What it does:**  
Runs the Johansen trace test to determine the cointegration rank *r* of the multivariate head-metric series.

```python
# analyzer.py — _check_cointegration()
result  = coint_johansen(series_np, det_order=0, k_ar_diff=k_diff)
rank    = int(np.sum(result.lr1 > result.cvt[:, 1]))   # trace stat > 95% CV
```

**Why it is appropriate:**  
If multiple attention heads share a common stochastic trend (i.e., they are cointegrated), first-differencing destroys the long-run relationship. The Johansen test detects this before model fitting, selecting VECM instead of plain VAR when r > 0. In reasoning terms: a positive cointegration rank indicates that a subset of heads has "locked on" to the same semantic target and is collectively converging toward a consistent answer.

---

### 3.10 Granger Causality and FDR Correction

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer._granger_pvalue_matrix()`, `_granger_pvalue_matrix_vecm()`, `_apply_fdr_correction()` |
| **Config** | `ChronoscopeConfig.granger_ftest = True`, `fdr_alpha = 0.05` |

**What it does:**  
For every directed pair (head j → head i), the Wald/F-test null hypothesis "Φ_{ij} = 0 for all lags" is tested. The H×H p-value matrix is corrected for multiple comparisons using Benjamini-Hochberg FDR:

```python
# analyzer.py — _granger_pvalue_matrix()
test = var_result.test_causality(caused=i, causing=j, kind='f')
pval_matrix[i, j] = test.pvalue

# analyzer.py — _apply_fdr_correction()
reject, pvals_corrected, _, _ = multipletests(flat_pvals, alpha=alpha, method='fdr_bh')
```

**Why it is appropriate:**  
- **Granger** detects *predictive influence*: head j Granger-causes head i if knowing j's past improves prediction of i's future beyond i's own past. This provides a directed influence map of the attention head network — identifying the "orchestrator" heads that drive collective reasoning.  
- **FDR correction** is essential: with H=14 heads there are H×(H−1) = 182 pairwise tests. Applying a naïve 0.05 threshold would yield ~9 false positives. BH-FDR controls the expected proportion of false discoveries, making the influence map statistically reliable.

---

### 3.11 Partial Directed Coherence (PDC)

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer._partial_directed_coherence()` |
| **Config** | `ChronoscopeConfig.compute_pdc = True` |

**What it does:**  
Derives the frequency-domain directed influence measure from VAR coefficients A_k:

> PDC(j→i, f) = |A(f)[i,j]|² / Σ_k |A(f)[k,j]|²

where A(f) = I − Σ_k A_k e^{−i2πfk} is the transfer matrix at frequency f.

```python
# analyzer.py — _partial_directed_coherence()
A_f_all = np.eye(H) - np.sum(coefs * exp_terms, axis=1)   # [F, H, H]
A_abs_sq = np.abs(A_f_all) ** 2
pdc = A_abs_sq / col_sums                                   # normalised
```

**Why it is appropriate:**  
PDC reveals not only *which* head drives another, but *at what reasoning frequency*. For example, a high PDC(j→i) at low frequency (f < 0.1) means head j shapes head i's slow, global context evolution; high PDC at high frequency means head j modulates rapid token-by-token adjustments. This frequency-resolved causal map cannot be obtained from time-domain Granger tests alone.

---

### 3.12 Transfer Entropy

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer._conditional_transfer_entropy()` |
| **Config** | `ChronoscopeConfig.use_transfer_entropy = False` (opt-in; computationally expensive) |

**What it does:**  
Computes conditional transfer entropy TE(j→i) = I(X_i(t+1) ; X_j^{1:k} | X_i^{1:k}) via symbol discretisation, providing a *model-free* (non-parametric) measure of directed information flow.

**Why it is appropriate:**  
VAR and Granger causality assume *linear* relationships. Transfer entropy makes no linearity assumption, detecting nonlinear head-to-head influence that VAR would miss. It serves as a robustness check: if Granger and TE agree on the same directed pairs, the influence is likely real rather than a linear artefact.

---

### 3.13 Bootstrap Phase-Scramble Surrogates

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer._bootstrap_surrogate_pvalues()` |
| **Config** | `ChronoscopeConfig.enable_bootstrap_surrogates`, `n_bootstrap_surrogates = 500` |

**What it does:**  
Generates surrogate series by randomising the phase of the FFT spectrum (preserving magnitude / power structure but destroying temporal order) using `torch.fft.rfft`, then refits VAR on each surrogate to build a null distribution of influence scores.

```python
# analyzer.py — _bootstrap_surrogate_pvalues()
spectrum   = torch.fft.rfft(series_t, dim=0)              # [F, H] complex
rand_phases = torch.rand(n_surrogates, F, H) * 2 * pi
scrambled  = magnitude * torch.polar(ones, rand_phases)   # [S, F, H]
surrogates = torch.fft.irfft(scrambled, n=T, dim=1)       # [S, T, H]
```

**Why it is appropriate:**  
Phase scrambling preserves the power spectrum (ruling out spectral artefacts) but destroys any temporal causality, creating a valid null distribution. Empirical p-values from this procedure are non-parametric and do not rely on the Gaussian residual assumptions of the parametric F-test, making them more reliable for the fat-tailed activation distributions common in LLMs.

---

### 3.14 Dynamic Time Warping (DTW)

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer.dtw_divergence()`, `CausalAnalyzer.soft_dtw_divergence()` |
| **Config** | `ChronoscopeConfig.dtw_radius = 5` (Sakoe-Chiba band) |

**What it does:**  
Computes the DTW distance (via `fastdtw` with cosine distance in SVD space) between the clean and activation-perturbed trajectories. `soft_dtw_divergence()` provides a differentiable PyTorch implementation using log-sum-exp smoothing.

```python
# analyzer.py — dtw_divergence()
distance, path = fastdtw(c_comp, p_comp, radius=config.dtw_radius, dist=cosine)
normalized = distance / len(path)
```

**Why it is appropriate:**  
Reasoning processes at different speeds: a model completing a long deductive chain generates many tokens before concluding, while a retrieval-based answer may conclude quickly. Euclidean or L2 distance applied frame-by-frame would falsely flag temporal *misalignment* as *semantic* divergence. DTW finds the optimal monotonic alignment, comparing the *content* of trajectories rather than their *timing*. The Sakoe-Chiba band (`radius=5`) limits the warp window, preventing pathological alignments and controlling computational cost.

---

### 3.15 Topological Data Analysis (TDA) — Persistent Homology

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer.topological_analysis()`, `CausalAnalyzer.topological_analysis_windowed()` |
| **Code** | `chronoscope/observer.py` → `SignalObserver.incremental_analysis()` (Euler Characteristic fast proxy) |
| **Config** | `ChronoscopeConfig.tda_enable_windowed = True`, `tda_window_size = 10` |

**What it does:**  
- **Full TDA**: runs `ripser` on the SVD-compressed trajectory, computing Vietoris-Rips persistence diagrams and Betti numbers (β₀ = connected components, β₁ = 1D loops). Extracts *Persistence Landscape L² norms* as a statistically stable topological summary.  
- **Euler Characteristic proxy**: real-time, O(n²) threshold-based graph construction computing χ = V − E from pairwise L2 distances for streaming token-by-token anomaly detection.

```python
# analyzer.py — topological_analysis()
result    = ripser.ripser(compressed, maxdim=1, thresh=np.inf)
diagrams  = result["dgms"]
lifetimes = finite[:, 1] - finite[:, 0]           # persistence = death - birth
l_norm    = sqrt(trapz(landscape**2, t_vals))      # Persistence Landscape L² norm

# observer.py — incremental_analysis()
adjacency = (distances < threshold).astype(int)
chi = V - E                                        # Euler Characteristic
```

**Why it is appropriate:**  
- **β₁ loops** detect when the reasoning trajectory *revisits* a region of latent space — the model is circling back to cross-reference a premise, a topological signature of recursive verification. Activation magnitudes or spectral methods cannot detect this geometric structure.  
- **Persistence Landscape** uses the Cohen-Steiner stability theorem: small perturbations in the trajectory produce bounded changes in the landscape norm. This makes TDA robust to the small numerical noise inherent in quantised activations.  
- **Euler Characteristic** (χ = V − E) is the simplest topological invariant and runs in real time without `ripser`. A sudden spike in χ during streaming generation flags an abrupt structural break — the model switching reasoning strategies mid-generation.

---

### 3.16 Composite Fidelity Score

| Attribute | Detail |
|-----------|--------|
| **Code** | `chronoscope/analyzer.py` → `CausalAnalyzer.compute_fidelity_score()` |

**What it does:**  
Aggregates all per-tool outputs into a single z-score-weighted *Fidelity Score* that maps to a reasoning quality verdict:

```python
# analyzer.py — compute_fidelity_score()
z_dtw         = (dtw_norm - 1.0) / 0.5              # semantic tracking
z_spectral    = (peak_ratio - 0.2) / 0.1            # spectral coherence
z_topology    = (h1_norm - 0.5) / 0.25              # TDA persistence
z_stationarity = (0.5 - variance_ratio) / 0.2       # active reasoning
# V2 extensions:
z_stability   = f(spectral_radius_of_VAR_roots)     # VAR dynamical stability
z_rank        = f(effective_rank_entropy)            # multi-head diversity
z_persistence = f(hmm_persistence_score)            # HMM contextual focus

validity_probability = norm.cdf(mean_z)
```

**Why it is appropriate:**  
Each component of the fidelity score captures a *different and complementary* aspect of reasoning quality:

| Z-score | What it captures | Tool |
|---------|-----------------|------|
| `z_dtw` | Semantic distance from clean baseline | DTW |
| `z_spectral` | Structured vs. random frequency content | Welch PSD |
| `z_topology` | Depth of recursive premise verification | TDA |
| `z_stationarity` | Active belief updating vs. plateau | ADF/Variance Ratio |
| `z_stability` | Stable vs. explosive head dynamics | VAR spectral radius |
| `z_rank` | Multi-head diversity (vs. collapsed single mode) | SVD effective rank |
| `z_persistence` | Contextual focus stability | HMM state persistence |

The composite is mapped through the standard normal CDF, producing a well-calibrated probability that reflects *all* available evidence simultaneously, avoiding over-reliance on any single heuristic.

---

## 4. Handling Statistical Assumptions for Deterministic LLM Signals

Classical time-series tools assume stochastic processes with random noise. LLM activations are, in theory, **deterministic** (same input → same output). Three techniques bridge this gap, all implemented in the codebase:

| Challenge | Solution | Code |
|-----------|----------|------|
| **Non-stochasticity** | Treat information-gain per token as the "innovation" ε_t in a state-space model | Conceptual framing in `Chronoscope_Math_Whitepaper.md`; variance-ratio stationarity test acts as empirical proxy |
| **Non-stationarity / drift** | ADF gatekeeper → first-differencing or VECM | `observer.py:stationarity_test()`, `analyzer.py:_apply_selective_differencing()`, `analyzer.py:_fit_vecm()` |
| **Fat-tailed / non-Gaussian activations** | Robust scaling (median/MAD) before VAR; phase-scramble bootstrap for non-parametric p-values; Persistence Landscapes (depend only on geometry) | `analyzer.py:_preprocess_series_for_var()` (`scale_mode='robust'`); `_bootstrap_surrogate_pvalues()`; `topological_analysis()` |

---

## 5. Summary Table

| Mathematical Tool | Code Location | Answers the Question |
|-------------------|---------------|----------------------|
| SVD compression | `observer.py:SignalObserver.svd_compress` | Which dimensions of hidden space carry the most reasoning variance? |
| Welch PSD / FFT | `observer.py:SignalObserver.spectral_analysis` | What is the dominant periodicity of reasoning substeps? |
| ACF / PACF | `observer.py:SignalObserver.autocorrelation_analysis` | How far back does the model's hidden state depend on its own past? |
| ADF + KPSS | `observer.py:stationarity_test`, `joint_stationarity_test` | Is the model actively updating beliefs (non-stationary) or plateaued? |
| Additive decomposition | `observer.py:SignalObserver.decompose` | What are the trend, periodic, and noise components of the reasoning path? |
| Velocity / Acceleration | `observer.py:SignalObserver.calculate_trajectory_dynamics` | When does the model transition rapidly between semantic states? |
| Hurst Exponent (DFA) | `observer.py:SignalObserver.calculate_trajectory_dynamics` | Is the model following a persistent logical path or a random walk? |
| Arc length / Curvature | `observer.py:SignalObserver.compute_intrinsic_time` | Which tokens represent the highest-effort reasoning steps? |
| VAR / VECM | `analyzer.py:CausalAnalyzer` | How do attention heads influence each other's future states? |
| Johansen cointegration | `analyzer.py:_check_cointegration` | Are multiple heads converging toward a common long-run belief? |
| Granger causality + FDR | `analyzer.py:_granger_pvalue_matrix`, `_apply_fdr_correction` | Which head-to-head directed influences are statistically significant? |
| PDC | `analyzer.py:_partial_directed_coherence` | At which reasoning frequencies does each head drive others? |
| Transfer Entropy | `analyzer.py:_conditional_transfer_entropy` | Are there nonlinear head-to-head influences that VAR misses? |
| Bootstrap surrogates | `analyzer.py:_bootstrap_surrogate_pvalues` | Are spectral and Granger findings significant vs. the phase-scrambled null? |
| DTW | `analyzer.py:CausalAnalyzer.dtw_divergence` | How semantically divergent is a perturbed trajectory from the clean baseline? |
| TDA / Persistent Homology | `analyzer.py:CausalAnalyzer.topological_analysis` | Does the reasoning path loop back (recursive verification) or flow forward? |
| Euler Characteristic (live) | `observer.py:SignalObserver.incremental_analysis` | Has the model experienced an abrupt structural break in real time? |
| Composite Fidelity Score | `analyzer.py:CausalAnalyzer.compute_fidelity_score` | What is the integrated probability that the model is reasoning soundly? |
