"""
Chronoscope Online Chat Server — Live model interaction with real-time analysis.

Architecture:
  :8000  — FastAPI chat API (POST /chat) + serves chat UI
  :8765  — DashboardBridge WebSocket (chronoscope_live.html connects here)
  :8766  — HTTP server for chronoscope_live.html dashboard

Flow:
  User prompt → POST /chat → model generates token-by-token →
  interceptor captures activations → observer analyses live →
  DashboardBridge pushes frames to dashboard → panels stream in real time

This is the ONLINE mode (always-on server, interactive chat).
Exp6 is OFFLINE mode (script, fixed prompt, exit).
"""

import asyncio
import json
import sys
import os
import webbrowser
from typing import Optional, Set
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Thread

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from chronoscope.config import ChronoscopeConfig
from chronoscope.models import load_model, get_deepest_layer, detect_num_attention_heads, detect_hidden_dim
from chronoscope.interceptor import ChronoscopeInterceptor
from chronoscope.observer import SignalObserver
from chronoscope.analyzer import CausalAnalyzer
from chronoscope.dashboard_bridge import DashboardBridge
from transformers import TextIteratorStreamer


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt text")
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    run_var: bool = Field(default=True, description="Run VAR head interaction analysis post-generation")
    run_intervention: bool = Field(default=True, description="Run head knockout intervention (top-3 heads, deepest layer)")


class MetricRequest(BaseModel):
    metric: str = Field(..., min_length=1, description="Metric key from dashboard tab")


class VarTuningRequest(BaseModel):
    fdr_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    var_preprocess_enable: Optional[bool] = None
    var_preprocess_logit_bounded: Optional[bool] = None
    var_preprocess_scale: Optional[str] = Field(default=None, pattern="^(none|zscore|robust)$")
    var_preprocess_remove_global_mean: Optional[bool] = None
    var_preprocess_clip: Optional[float] = None


_UI_TO_METRIC = {
    "shannon": "shannon_entropy",
    "renyi": "renyi_entropy_2",
    "effrank": "effective_rank",
    "sink": "sink_fraction",
    "maxattn": "max_attention",
}


from pydantic import BaseModel, Field, model_validator

def _sanitize_numpy(obj):
    """Recursively strip NumPy types to native Python types for clean JSON serialization."""
    if isinstance(obj, (np.bool_, np.generic)) and hasattr(obj, 'item'):
        # handles np.bool_, np.float32, np.int64, etc.
        item = obj.item()
        # Ensure we return native bool/int/float, since numpy scalar .item() does this
        return item
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "tolist"):
        # torch.Tensor support for debug endpoints
        return _sanitize_numpy(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return [_sanitize_numpy(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, set):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    return obj

class ChatResponse(BaseModel):
    status: str
    prompt: str = ""
    response: str = ""
    tokens_generated: int = 0
    analysis_summary: dict = {}
    message: str = ""

    @model_validator(mode='after')
    def sanitize_types(self):
        self.analysis_summary = _sanitize_numpy(self.analysis_summary)
        return self


# ============================================================================
# GLOBAL STATE
# ============================================================================

_model = None
_tokenizer = None
_config: Optional[ChronoscopeConfig] = None
_interceptor: Optional[ChronoscopeInterceptor] = None
_observer: Optional[SignalObserver] = None
_analyzer = None
_bridge: Optional[DashboardBridge] = None
_generation_lock = asyncio.Lock()


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_system():
    """Load model + analysis stack + start DashboardBridge on :8765."""
    global _model, _tokenizer, _config, _interceptor, _observer, _analyzer, _bridge

    print("=" * 60)
    print("CHRONOSCOPE ONLINE — Live Chat + Analysis Dashboard")
    print("=" * 60)

    # ── Config ────────────────────────────────────────────────────────
    _config = ChronoscopeConfig()
    # Live dashboard should capture all per-head metric features in parallel.
    _config.head_feature_mode = "vector"
    print(f"  Model:  {_config.model_name}")
    print(f"  Device: {_config.device}")
    print(f"  Head metric mode: {_config.head_feature_mode}")

    # ── Load model ────────────────────────────────────────────────────
    _model, _tokenizer = load_model(_config)
    print("[OK] Model loaded")

    # ── Auto-detect model architecture parameters ─────────────────────
    detected_heads = detect_num_attention_heads(_model)
    detected_dim = detect_hidden_dim(_model)
    
    if detected_heads and detected_heads > 0:
        print(f"  [AUTO] Detected {detected_heads} attention heads (configured: {_config.n_heads})")
        _config.n_heads = detected_heads
    
    if detected_dim and detected_dim > 0:
        print(f"  [AUTO] Detected {detected_dim} hidden dimension (configured: {_config.hidden_dim})")
        _config.hidden_dim = detected_dim

    # ── Analysis components ───────────────────────────────────────────
    _interceptor = ChronoscopeInterceptor(_model, _tokenizer, _config)
    _observer = SignalObserver(_config)
    _analyzer = CausalAnalyzer(_interceptor, _observer, _config)
    print("[OK] Interceptor + Observer + Analyzer ready")

    # ── DashboardBridge on :8765 (what chronoscope_live.html expects) ──
    try:
        _bridge = DashboardBridge(
            transport=getattr(_config, "dashboard_transport", "websocket"),
            ws_port=int(getattr(_config, "dashboard_ws_port", 8765)),
        ).start()

        dashboard_url = _bridge.serve_dashboard(
            getattr(_config, "dashboard_html_path",
                    "integration_hub/frontend/public/chronoscope_live.html"),
            port=int(getattr(_config, "dashboard_http_port", 8766)),
        )
        print(f"[OK] Dashboard: {dashboard_url}")
        print(f"[OK] Dashboard WS: ws://localhost:{_config.dashboard_ws_port}")
        _bridge.push_log("ok", f"Online server starting · model={_config.model_name}")

        # Auto-open dashboard in browser
        webbrowser.open(dashboard_url)
    except Exception as e:
        print(f"[!] Dashboard bridge failed: {e}")
        _bridge = None

    print(f"\n[OK] Chat API ready at http://127.0.0.1:8000/chat")
    print(f"     POST a JSON body: {{\"prompt\": \"your question\"}}\n")


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_system()
    yield
    if _interceptor:
        _interceptor.cleanup()
    if _bridge:
        _bridge.push_log("ok", "Server shutting down")
        _bridge.stop()


app = FastAPI(title="Chronoscope Online", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# CHAT ENDPOINT — Online analysis while model reasons
# ============================================================================

def run_post_analysis_background(target_layer, current_traj, prompt_len, token_count, prompt, run_var, run_intervention):
    """
    Runs the heavy STATSMODELS and CPU-bound topological analysis asynchronously.
    Updates the dashboard via the bridge once finished.
    """
    global _observer, _analyzer, _bridge, _config
    
    print("[*] Background: Running post-generation analysis...")
    if _bridge:
        _bridge.push_log("ok", f"Generation complete ({token_count} tokens). Running deep analysis...")
    fdr_sig_pairs = None

    try:
        if target_layer and target_layer in current_traj:
            traj_tensor = current_traj[target_layer]
            gen_tensor = traj_tensor[prompt_len:] if traj_tensor.shape[0] > prompt_len else traj_tensor

            if gen_tensor.shape[0] >= 5:
                # Full observer analysis
                observer_results = _observer.full_analysis(gen_tensor)

                print(f"  Hurst={_observer.hurst_exponent:.3f}, "
                      f"tau={_observer.tau_normalised:.3f}, "
                      f"ADF p={_observer.adf_pval_pc0:.4f}")

                if _bridge:
                    _bridge.push_token_frame(token_count, _interceptor, _observer, _config)

                # TDA
                compressed = observer_results["compressed_trajectory"]
                tda_results = _analyzer.topological_analysis(compressed)
                betti = tda_results.get("betti_numbers", {})

                if _bridge:
                    _bridge.push_tda_frame(
                        tda_result={
                            "betti0": int(betti.get("betti_0", 0)),
                            "betti1": int(betti.get("betti_1", 0)),
                            "euler": int(betti.get("betti_0", 0) - betti.get("betti_1", 0)),
                            "ec_series": _observer._ec_history,
                            "anomalies": [],
                        },
                        current_token=token_count,
                        phase_boundaries=_observer.phase_boundaries,
                        current_phase_idx=_observer.current_phase_idx,
                    )

                # VAR head interaction analysis (if enabled)
                if run_var and getattr(_config, "capture_attentions", False):
                    print("[*] Background: Running VAR head interaction analysis (multi-layer)...")
                    head_result = _analyzer.head_interaction_analysis(
                        prompt  # No layer_name → analyzer auto-selects mid + deepest layers
                    )
                    if "error" not in head_result:
                        fdr_sig_pairs = int((head_result.get("fdr_result") or {}).get("n_significant", 0))
                        if _bridge:
                            _bridge.push_var_frame(head_result)
                            _bridge.push_log("ok", "VAR analysis complete")

                        # ── D.4: HMM Phase Discovery ──────────────────────────
                        if getattr(_config, "use_hmm_phase_discovery", False):
                            try:
                                series_for_hmm = head_result.get("series")
                                if series_for_hmm is not None and series_for_hmm.shape[0] >= 10:
                                    hmm_result = _analyzer._discover_phases_hmm(
                                        series_for_hmm,
                                        n_states=getattr(_config, "hmm_n_states", 4),
                                    )
                                    if "error" not in hmm_result and _bridge:
                                        _bridge.push_hmm_frame(hmm_result, _config)
                                        _bridge.push_log("ok", "HMM phases fitted")
                            except Exception as hmm_e:
                                print(f"[!] HMM failed: {hmm_e}")
                                if _bridge:
                                    _bridge.push_log("err", f"HMM: {str(hmm_e)[:60]}")

                        # ── Live Intervention (Knockout Probe) ──
                        if run_intervention:
                            print("[*] Background: Running live intervention...")
                            if _bridge:
                                _bridge.push_log("pert", "Running live head interventions...")

                            fdr_res = head_result.get("fdr_result") or {}
                            sig_pairs = fdr_res.get("significant_pairs", [])
                            source_heads = []
                            if sig_pairs:
                                # Primary path: top FDR-significant source heads.
                                source_heads = list(dict.fromkeys(
                                    int(p[0]) for p in sig_pairs
                                ))[:3]
                            else:
                                # Demo fallback: if FDR is empty, probe strongest raw-influence
                                # sources so the perturbation panel can still provide signal.
                                influence = head_result.get("influence_matrix")
                                try:
                                    inf = np.asarray(influence, dtype=float)
                                    if inf.ndim == 2 and inf.shape[0] == inf.shape[1] and inf.shape[0] >= 2:
                                        score = inf.copy()
                                        np.fill_diagonal(score, -np.inf)
                                        flat_order = np.argsort(score, axis=None)[::-1]
                                        for flat_idx in flat_order:
                                            i, j = np.unravel_index(flat_idx, score.shape)
                                            v = score[i, j]
                                            if not np.isfinite(v) or v <= 0.0:
                                                continue
                                            if int(j) not in source_heads:
                                                source_heads.append(int(j))
                                            if len(source_heads) >= 3:
                                                break
                                        if source_heads and _bridge:
                                            _bridge.push_log(
                                                "pert",
                                                "No FDR-significant pairs; using top influence sources for intervention demo.",
                                            )
                                except Exception:
                                    source_heads = []

                            if source_heads:
                                # Use target_layer if available, else fall back to midpoint layer
                                probe_layers = []
                                if target_layer:
                                    probe_layers = [target_layer]
                                else:
                                    # Fallback: use deepest available layer from activation dict
                                    if current_traj and current_traj.keys():
                                        fallback_layer = get_deepest_layer(current_traj.keys())
                                        if fallback_layer:
                                            probe_layers = [fallback_layer]
                                            print(f"[*] No target_layer, using fallback: {fallback_layer}")

                                if probe_layers:
                                    try:
                                        causal_impact = _analyzer.interventional_head_causality_multilayer(
                                            prompt, probe_layers, source_heads
                                        )
                                        per_head = causal_impact.get("per_head_results", [])

                                        if per_head and _bridge:
                                            pert_results = [
                                                {
                                                    "head":          int(r["head"]),
                                                    "target":        int(r.get("target_head", -1)),
                                                    "mode":          "zero",
                                                    "delta_entropy": float(r.get("delta_entropy", 0.0)),
                                                    "restoration":   float(r.get("restoration", 0.0)),
                                                    "kl_patch":      float(r.get("kl_patch", 0.0)),
                                                    "confirmed":     float(r.get("restoration", 0.0)) > 0.5,
                                                }
                                                for r in per_head
                                            ]
                                            _bridge.push_perturbation_frame(pert_results, None)
                                            _bridge.push_log("ok", f"Interventions complete on {len(pert_results)} heads")
                                        else:
                                            print(f"[!] No intervention results returned (per_head empty)")
                                            if _bridge:
                                                _bridge.push_log("pert", f"Intervention analysis yielded no results")
                                    except Exception as pert_err:
                                        print(f"[!] Intervention error: {pert_err}")
                                        if _bridge:
                                            _bridge.push_log("err", f"Intervention failed: {str(pert_err)[:60]}")
                                else:
                                    print(f"[!] No probe layers available for intervention")
                                    if _bridge:
                                        _bridge.push_log("pert", "No target layer for intervention")
                            else:
                                if _bridge:
                                    _bridge.push_log("pert", "No usable source heads found — skipping intervention.")
                    else:
                        if _bridge:
                            _bridge.push_log("err", f"VAR: {head_result['error']}")

                # Validity score
                validity = _analyzer.compute_validity_score(
                    dtw_result={},
                    spectral_result=observer_results.get("spectral", {}),
                    tda_result=tda_results,
                    stationarity_result=observer_results.get("stationarity", {}),
                )

                composite_score = int(round(validity.get("composite_validity", 0) * 100))
                verdict = (
                    "STRONG REASONING" if composite_score > 70 else
                    "MODERATE REASONING" if composite_score > 50 else
                    "HALLUCINATION RISK"
                )

                if _bridge:
                    def _sf(v):
                        try: return float(v) if v is not None else None
                        except (TypeError, ValueError): return None

                    _bridge.push_score_frame(
                        {
                            "score": composite_score,
                            "verdict": verdict,
                            "dtw_sensitivity": _sf(validity.get("dtw_sensitivity")),
                            "spectral_coherence": _sf(validity.get("spectral_validity")),
                            "topo_smoothness": _sf(validity.get("tda_validity")),
                            "active_reasoning": _sf(validity.get("active_reasoning")),
                            "fdr_sig_pairs": fdr_sig_pairs,
                            "te_score": None,
                        },
                        {
                            "source": "Background Analysis",
                            "text": f"Score={composite_score}/100 · {verdict}",
                        },
                    )
    except Exception as e:
        print(f"[!] Background analysis error: {e}")
        import traceback
        traceback.print_exc()
        if _bridge:
            _bridge.push_log("err", f"Deep analysis failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Accepts a prompt, generates token-by-token, and streams live
    Chronoscope analysis to the dashboard as the model reasons.
    """
    global _model, _tokenizer, _config, _interceptor, _observer, _analyzer, _bridge

    if not _model or not _tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if _generation_lock.locked():
        raise HTTPException(status_code=429, detail="Generation in progress — wait for current request")

    async with _generation_lock:
        prompt = request.prompt.strip()
        max_tokens = request.max_tokens
        temperature = request.temperature

        print(f"\n[CHAT] Prompt: {prompt[:80]}...")
        if _bridge:
            _bridge.push_log("ok", f"New prompt: {prompt[:50]}...")

        # ── Tokenize + measure prompt length ─────────────────────────
        inputs = _tokenizer(prompt, return_tensors="pt").to(_config.device)
        prompt_len = inputs.input_ids.shape[1]

        # ── Clear interceptor state ──────────────────────────────────
        _interceptor._clear()
        _observer.__init__(_config)  # Reset observer for fresh analysis

        # ── Stream generation with live analysis ─────────────────────
        full_response = ""
        token_count = 0
        token_count_by_text = 0
        analysis_summary = {}

        try:
            # Use capture_generation_stream for live token-by-token analysis
            # This is the same pattern as exp6 capture_node (E3-K1)
            print("[*] Streaming: ", end="", flush=True)

            for new_token, current_traj in _interceptor.capture_generation_stream(
                prompt, max_new_tokens=max_tokens
            ):
                sys.stdout.write(new_token)
                sys.stdout.flush()
                full_response += new_token
                try:
                    # Streamers can emit multi-token text chunks. Count by
                    # tokenizer delta over the cumulative generated text.
                    current_count = len(_tokenizer.encode(full_response, add_special_tokens=False))
                    delta = max(0, current_count - token_count_by_text)
                    token_count_by_text = current_count
                    token_count += delta if delta > 0 else 1
                except Exception:
                    token_count += 1

                # ── Live analysis per token ──────────────────────────
                target_layer = get_deepest_layer(current_traj.keys()) if current_traj else None

                if target_layer and target_layer in current_traj:
                    traj = current_traj[target_layer]
                    gen_only = traj[prompt_len:] if traj.shape[0] > prompt_len else traj

                    if gen_only.shape[0] >= 2:
                        # Live TDA + EC tracking (E4-K1)
                        live_stats = _observer.incremental_analysis(
                            gen_only,
                            window_size=getattr(_config, "tda_window_size", 10),
                            distance_threshold=getattr(_config, "tda_distance_threshold", 2.0),
                        )

                        # Push TDA anomaly if detected (E3-K2)
                        if live_stats.get("topological_anomaly_detected") and _bridge:
                            _bridge.push_tda_frame(
                                tda_result={
                                    "betti0": live_stats.get("betti0"),
                                    "betti1": live_stats.get("betti1"),
                                    "euler": live_stats.get("euler_characteristic"),
                                    "ec_series": live_stats.get("ec_series", []),
                                    "anomalies": [{
                                        "token_idx": token_count,
                                        "severity": "high",
                                        "description": live_stats.get("diagnostics", "EC spike"),
                                    }],
                                },
                                current_token=token_count,
                            )
                            _bridge.push_log("tda",
                                f"D.2: EC spike t={token_count} · "
                                f"χ={live_stats.get('euler_characteristic', 0)}")

                # Push per-token entropy frame to dashboard (E3-K1)
                # Do this even when current_traj is empty for a streamer chunk;
                # bridge can pull latest metric row from interceptor cache.
                if _bridge:
                    row = None
                    if target_layer:
                        metric_series = _interceptor.get_head_metric_series(target_layer)
                        if metric_series is not None and getattr(metric_series, "numel", lambda: 0)() > 0:
                            row = metric_series[-1]
                    _bridge.push_token_frame(
                        token_idx=token_count,
                        interceptor=_interceptor,
                        observer=_observer,
                        config=_config,
                        entropy_row_override=row,
                    )

                # Cadenced signal quality (E2L-K1)
                stat_every = getattr(_config, "dashboard_stat_every", 5)
                if token_count % stat_every == 0 and _bridge:
                    _bridge.push_signal_quality_frame(_interceptor, _config)

                # Yield control for async responsiveness
                await asyncio.sleep(0)

            print(f"\n[OK] Generated {token_count} tokens")

        except Exception as e:
            print(f"\n[!] Generation error: {e}")
            import traceback
            traceback.print_exc()
            if _bridge:
                _bridge.push_log("err", f"Generation error: {str(e)[:80]}")
            return ChatResponse(
                status="error",
                message=str(e),
                response=full_response,
                tokens_generated=token_count,
            )

        # ── Schedule post-generation deep analysis in the background ──
        print("[*] Scheduling post-generation analysis...")
        target_layer = get_deepest_layer(current_traj.keys()) if current_traj else None
        
        # Schedule it
        background_tasks.add_task(
            run_post_analysis_background,
            target_layer=target_layer,
            current_traj=current_traj,
            prompt_len=prompt_len,
            token_count=token_count,
            prompt=prompt,
            run_var=request.run_var,
            run_intervention=request.run_intervention
        )
        
        # Quick summary for instant response
        analysis_summary = {
            "status": "Analysis computing in background...",
        }

        return ChatResponse(
            status="success",
            prompt=prompt,
            response=full_response,
            tokens_generated=token_count,
            analysis_summary=analysis_summary,
        )


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "bridge_active": _bridge is not None,
    }


@app.post("/set_metric")
async def set_metric(request: MetricRequest):
    if not _config:
        raise HTTPException(status_code=503, detail="Not initialized")

    ui_metric = request.metric.strip().lower()
    metric_key = _UI_TO_METRIC.get(ui_metric)
    if metric_key is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric '{request.metric}'. Allowed: {', '.join(_UI_TO_METRIC.keys())}",
        )

    _config.head_metric_type = metric_key
    return {
        "status": "ok",
        "ui_metric": ui_metric,
        "head_metric_type": _config.head_metric_type,
        "head_feature_mode": getattr(_config, "head_feature_mode", "scalar"),
    }


@app.post("/set_var_tuning")
async def set_var_tuning(request: VarTuningRequest):
    if not _config:
        raise HTTPException(status_code=503, detail="Not initialized")

    updates = {}

    if request.fdr_alpha is not None:
        _config.fdr_alpha = float(request.fdr_alpha)
        updates["fdr_alpha"] = _config.fdr_alpha
    if request.var_preprocess_enable is not None:
        _config.var_preprocess_enable = bool(request.var_preprocess_enable)
        updates["var_preprocess_enable"] = _config.var_preprocess_enable
    if request.var_preprocess_logit_bounded is not None:
        _config.var_preprocess_logit_bounded = bool(request.var_preprocess_logit_bounded)
        updates["var_preprocess_logit_bounded"] = _config.var_preprocess_logit_bounded
    if request.var_preprocess_scale is not None:
        _config.var_preprocess_scale = request.var_preprocess_scale
        updates["var_preprocess_scale"] = _config.var_preprocess_scale
    if request.var_preprocess_remove_global_mean is not None:
        _config.var_preprocess_remove_global_mean = bool(request.var_preprocess_remove_global_mean)
        updates["var_preprocess_remove_global_mean"] = _config.var_preprocess_remove_global_mean
    if request.var_preprocess_clip is not None:
        _config.var_preprocess_clip = float(request.var_preprocess_clip)
        updates["var_preprocess_clip"] = _config.var_preprocess_clip

    return {
        "status": "ok",
        "updated": updates,
        "effective": {
            "fdr_alpha": getattr(_config, "fdr_alpha", None),
            "var_preprocess_enable": getattr(_config, "var_preprocess_enable", None),
            "var_preprocess_logit_bounded": getattr(_config, "var_preprocess_logit_bounded", None),
            "var_preprocess_scale": getattr(_config, "var_preprocess_scale", None),
            "var_preprocess_remove_global_mean": getattr(_config, "var_preprocess_remove_global_mean", None),
            "var_preprocess_clip": getattr(_config, "var_preprocess_clip", None),
        },
    }


@app.get("/debug/last_frame")
async def debug_last_frame():
    if not _bridge:
        raise HTTPException(status_code=503, detail="Bridge not initialized")
    frame = getattr(_bridge, "_last_frame", None) or {}
    return {
        "status": "ok",
        "keys": sorted(list(frame.keys())),
        "frame": _sanitize_numpy(frame),
    }


@app.get("/config")
async def get_config():
    if not _config:
        raise HTTPException(status_code=503, detail="Not initialized")
    return {
        "model_name": _config.model_name,
        "device": _config.device,
        "n_heads": _config.n_heads,
        "hidden_dim": _config.hidden_dim,
        "target_layer": _config.target_layer,
        "head_feature_mode": getattr(_config, "head_feature_mode", "scalar"),
        "head_metric_type": getattr(_config, "head_metric_type", "shannon_entropy"),
    }


@app.get("/debug/metrics")
async def debug_metrics():
    """Inspect what head metrics have been captured by the interceptor."""
    if not _interceptor:
        raise HTTPException(status_code=503, detail="Interceptor not initialized")
    
    summary = _interceptor.debug_head_metric_summary()
    
    # Extract first timestep metrics for each layer for inspection
    details = {}
    for layer_key, series_list in _interceptor._head_metrics.items():
        details[layer_key] = {
            "num_timesteps": len(series_list),
            "first_shape": str(series_list[0].shape) if series_list else "empty",
            "last_shape": str(series_list[-1].shape) if series_list else "empty",
        }
    
    # Also check what the latest frame tried to find
    last_frame = getattr(_bridge, "_last_frame", {}) if _bridge else {}
    
    return {
        "status": "ok",
        "capture_attentions_config": _config.capture_attentions,
        "metrics_summary": summary,
        "metrics_details": details,
        "last_frame_has_entropy_row": "entropy_row" in last_frame,
        "last_frame_entropy_row_is_none": last_frame.get("entropy_row") is None,
    }


@app.get("/debug/perturbation")
async def debug_perturbation():
    """Check perturbation/ablation (Gap C) state in last frame."""
    if not _bridge:
        raise HTTPException(status_code=503, detail="Bridge not initialized")
    
    last_frame = getattr(_bridge, "_last_frame", {}) or {}
    
    return {
        "status": "ok",
        "has_perturbation_results": "perturbation_results" in last_frame,
        "perturbation_count": len(last_frame.get("perturbation_results", [])),
        "first_pert": last_frame.get("perturbation_results", [{}])[0] if last_frame.get("perturbation_results") else None,
        "mediation_results_present": last_frame.get("mediation_results") is not None,
        "last_frame_keys": list(last_frame.keys()),
    }


@app.get("/debug/websocket")
async def debug_websocket():
    """Check WebSocket server status and connected clients."""
    if not _bridge:
        raise HTTPException(status_code=503, detail="Bridge not initialized")
    
    ws_clients = getattr(_bridge, "_ws_clients", set())
    ws_thread = getattr(_bridge, "_thread", None)
    
    return {
        "status": "ok",
        "websocket_port": getattr(_bridge, "ws_port", 8765),
        "websocket_host": getattr(_bridge, "ws_host", "127.0.0.1"),
        "connected_clients": len(ws_clients),
        "ws_thread_alive": ws_thread.is_alive() if ws_thread else False,
        "has_last_frame": getattr(_bridge, "_last_frame", None) is not None,
        "last_frame_keys": sorted(list(getattr(_bridge, "_last_frame", {}).keys())),
    }


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Serve a minimal chat interface that POSTs to /chat."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chronoscope Chat</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:#0a0a0f; color:#e0e0e0; height:100vh; display:flex; flex-direction:column; }
  header { background:linear-gradient(135deg,#1a1a2e,#16213e); padding:16px 24px; border-bottom:1px solid #ffffff15; }
  header h1 { font-size:1.2em; background:linear-gradient(90deg,#00d4ff,#00ff88); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  header p { font-size:0.8em; color:#888; margin-top:4px; }
  #messages { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:80%; padding:12px 16px; border-radius:12px; font-size:0.95em; line-height:1.5; }
  .msg.user { align-self:flex-end; background:#1e3a5f; border:1px solid #2a5a8f; }
  .msg.assistant { align-self:flex-start; background:#1a1a2e; border:1px solid #ffffff15; }
  .msg .meta { font-size:0.75em; color:#666; margin-top:6px; }
  .msg .verdict { font-weight:bold; padding:2px 8px; border-radius:4px; font-size:0.8em; }
  .verdict.strong { background:#00ff8830; color:#00ff88; }
  .verdict.moderate { background:#ffb30030; color:#ffb300; }
  .verdict.risk { background:#ff3d5030; color:#ff3d50; }
  #input-bar { display:flex; gap:10px; padding:16px 24px; background:#0d0d15; border-top:1px solid #ffffff10; }
  #prompt { flex:1; background:#1a1a2e; border:1px solid #ffffff20; color:#fff; padding:12px 16px; border-radius:8px; font-size:1em; outline:none; }
  #prompt:focus { border-color:#00d4ff; }
  #send { background:linear-gradient(135deg,#00d4ff,#00a8cc); color:#000; font-weight:bold; border:none; padding:12px 24px; border-radius:8px; cursor:pointer; }
  #send:disabled { opacity:0.4; cursor:not-allowed; }
  .spinner { display:inline-block; width:16px; height:16px; border:2px solid #00d4ff40; border-top-color:#00d4ff; border-radius:50%; animation:spin .6s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  a { color:#00d4ff; }
</style>
</head>
<body>
<header>
  <h1>⟐ Chronoscope Online</h1>
  <p>Live chat with real-time analysis → <a href="http://localhost:8766" target="_blank">Open Dashboard</a></p>
</header>
<div id="messages">
  <div class="msg assistant">Ready. Type a prompt and watch the <a href="http://localhost:8766" target="_blank">live dashboard</a> as the model reasons.</div>
</div>
<div id="input-bar">
  <input id="prompt" type="text" placeholder="Ask something..." autofocus />
  <button id="send" onclick="sendChat()">Send</button>
</div>
<script>
const msgs = document.getElementById('messages');
const inp = document.getElementById('prompt');
const btn = document.getElementById('send');

inp.addEventListener('keydown', e => { if(e.key==='Enter' && !btn.disabled) sendChat(); });

async function sendChat() {
  const prompt = inp.value.trim();
  if(!prompt) return;
  inp.value = '';
  btn.disabled = true;

  // User message
  const um = document.createElement('div');
  um.className = 'msg user';
  um.textContent = prompt;
  msgs.appendChild(um);

  // Thinking indicator
  const tm = document.createElement('div');
  tm.className = 'msg assistant';
  tm.innerHTML = '<span class="spinner"></span> Generating + analyzing...';
  msgs.appendChild(tm);
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt, max_tokens:150, run_var:true})
    });
    const data = await res.json();

    let verdictHtml = '';
    if(data.analysis_summary && data.analysis_summary.verdict) {
      const v = data.analysis_summary.verdict;
      const cls = v.includes('STRONG') ? 'strong' : v.includes('MODERATE') ? 'moderate' : 'risk';
      verdictHtml = '<br><span class="verdict '+cls+'">'+v+' ('+data.analysis_summary.composite_score+'/100)</span>';
    }

    let metaHtml = '';
    if(data.analysis_summary) {
      const s = data.analysis_summary;
      const parts = [];
      if(s.hurst !== undefined) parts.push('H='+s.hurst.toFixed(3));
      if(s.tau !== undefined) parts.push('τ='+s.tau.toFixed(3));
      if(s.is_stationary !== undefined) parts.push(s.is_stationary ? 'Stationary' : 'Non-stationary');
      if(parts.length) metaHtml = '<div class="meta">'+parts.join(' · ')+'</div>';
    }

    tm.innerHTML = data.response + verdictHtml + metaHtml;
  } catch(err) {
    tm.innerHTML = '<span style="color:#ff3d50">Error: '+err.message+'</span>';
  }
  btn.disabled = false;
  msgs.scrollTop = msgs.scrollHeight;
}
</script>
</body>
</html>"""


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
