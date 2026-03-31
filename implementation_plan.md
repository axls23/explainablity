# Implementation Plan - Fix Critical Issues

This plan addresses 6 critical issues (3 High, 3 Medium) identified in the Chronoscope system, spanning the statistical analyzer, dashboard bridge, and frontend visualization.

## User Review Required

> [!IMPORTANT]
> The vectorization of PDC (Partial Directed Coherence) will significantly improve performance but requires `np.exp` broadcasting across frequency and lag dimensions. This assumes the frequency range is reasonably dense (default 128).

> [!WARNING]
> The Granger test for VECM (Vector Error Correction Model) requires using the `gamma` (short-run coefficients) and ensuring the degrees of freedom are calculated correctly for the cointegrated system.

## Proposed Changes

### Chronoscope Analyzer (`chronoscope/analyzer.py`)

#### [MODIFY] [analyzer.py](file:///c:/dev/explainablity/chronoscope/analyzer.py)

1.  **Fix VAR dimension check (High Priority)**
    *   **Location**: Line 1356
    *   **Change**: Change `T // 4` to `T // 5` to increase the safety margin for short sequences, preventing rank-deficiency or overfitting in small samples.

2.  **Vectorize PDC frequency loop (High Priority)**
    *   **Location**: Lines 881-889 in `_partial_directed_coherence`
    *   **Change**: Replace the explicit loop over frequencies with vectorized NumPy operations.
    *   **Logic**:
        *   Create a complex exponential tensor `exp_tensor = np.exp(-2j * np.pi * freqs[:, None] * (np.arange(p) + 1))` with shape `(F, p)`.
        *   Contract this with `coefs` `(p, H, H)` to get `A(f)` `(F, H, H)` in one go using `np.einsum`.

3.  **HMM BIC formula (Medium Priority)**
    *   **Location**: Line 1234
    *   **Change**: Update the parameter count `n_params` to handle potentially full covariance matrices, even if currently set to 'diag'.
    *   **Fix**: `n_params = n_states * (n_states - 1) + n_states * H + (n_states * H if model.covariance_type == 'diag' else n_states * H * (H + 1) // 2) + (n_states - 1)` (or similar adjusted formula).

4.  **Granger test for VECM (Medium Priority)**
    *   **Location**: Line 1510
    *   **Change**: Allow the Granger block to run if `vecm_result` is present.
    *   **Logic**: If `USE_VECM` and `vecm_result` exist, use `vecm_result` for the Granger F-test (focusing on short-run effects).

### Dashboard Bridge (`chronoscope/dashboard_bridge.py`)

#### [MODIFY] [dashboard_bridge.py](file:///c:/dev/explainablity/chronoscope/dashboard_bridge.py)

1.  **Fix shutdown race in WebSocket broadcast (High Priority)**
    *   **Location**: Line 614 in `_ws_broadcast`
    *   **Change**: Add a check to ensure the event loop is running before attempting to schedule a coroutine.
    *   **Fix**: Check `self._loop.is_running()` or use a `self._running` flag set in `start()` and cleared in `stop()`.

### Chronoscope Live Dashboard (`integration_hub/frontend/public/chronoscope_live.html`)

#### [MODIFY] [chronoscope_live.html](file:///c:/dev/explainablity/integration_hub/frontend/public/chronoscope_live.html)

1.  **Dynamic Y-axis for Entropy Chart (Medium Priority)**
    *   **Location**: Line 935 in `redrawEntropyChartForMetric`
    *   **Change**: Update the `yMaxByMetric` map to include more metrics and ensure consistent scaling for metrics like Gini, Kurtosis, etc.
    *   **Fix**: Expand the map and verify scales for newly added metrics in the backend (from `analyzer.py:1312-1324`).

## Open Questions

- For the HMM BIC fix, should I also change the `covariance_type` to 'full' by default, or just make the BIC formula aware of it?
- For the Granger/VECM fix, should I use the `vecm_result.test_granger_causality()` provided by statsmodels, or continue with the manual F-test approach for consistency?

## Verification Plan

### Automated Tests
- Run `pytest` if available.
- Specifically test `test_var.py` to ensure the VAR/VECM logic remains sound.
- Run `test_metrics_streaming.py` to verify the bridge.

### Manual Verification
- Launch the analyzer with a short prompt to trigger the `T // 5` logic.
- Open `chronoscope_live.html` and switch between different metric tabs (Shannon, Sink, etc.) to verify Y-axis scaling.
- Exit the analyzer and verify the terminal doesn't hang (shutdown race fix).
