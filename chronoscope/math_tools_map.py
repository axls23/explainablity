"""
Mathematical Tools Map for Chronoscope.

Provides a structured, programmatically-queryable mapping from mathematical
tool categories to the concrete functions in the Chronoscope codebase that
implement them.  Each entry explains the function's role and why it is
relevant to the project's core objective: modeling LLM internal activities
as a multivariate time series.

Categories
----------
1. Spectral Analysis / FFT
2. Statistical Features
3. Autocorrelation / Lag Analysis
4. Dimensionality Reduction
5. Change-point / State Detection
6. Evaluation / Validation Utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionEntry:
    """
    Describes a single function that implements (or directly supports) a
    mathematical tool used to model LLM internal activities as time series.

    Attributes
    ----------
    module : str
        Dot-separated module path relative to the ``chronoscope`` package
        (e.g. ``"chronoscope.observer"``).
    class_name : str
        Name of the class that owns the method (empty string for stand-alone
        functions).
    function : str
        Name of the function or method.
    mathematical_basis : str
        The underlying mathematical construct (e.g. ``"Welch PSD, FFT"``).
    role : str
        One-sentence description of what the function does in the pipeline.
    justification : str
        Why this function is relevant to modeling LLM internal activities
        as a time series and to the project's explainability objective.
    """

    module: str
    class_name: str
    function: str
    mathematical_basis: str
    role: str
    justification: str

    @property
    def qualified_name(self) -> str:
        """Return a fully-qualified ``module.Class.function`` name."""
        if self.class_name:
            return f"{self.module}.{self.class_name}.{self.function}"
        return f"{self.module}.{self.function}"


# ---------------------------------------------------------------------------
# Mapping definition
# ---------------------------------------------------------------------------

#: Complete code-grounded mapping from tool category to function entries.
MATH_TOOLS_MAP: Dict[str, List[FunctionEntry]] = {

    # ── 1. Spectral Analysis / FFT ─────────────────────────────────────── #
    "spectral_analysis_fft": [
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="spectral_analysis",
            mathematical_basis="Welch power spectral density (PSD) via scipy.signal.welch; "
                               "falls back to periodogram for very short sequences",
            role="Runs Welch's method on each SVD component of the hidden-state "
                 "trajectory to extract dominant frequencies and their power.",
            justification=(
                "LLM activations, viewed token-by-token, often contain quasi-periodic "
                "structure linked to sentence rhythm, attention patterns, and reasoning "
                "cycles.  Spectral analysis converts the token-indexed residual stream "
                "into the frequency domain, making these recurring patterns directly "
                "observable and quantifiable (dominant periods, spectral energy "
                "concentration).  Welch's method was preferred over a raw periodogram "
                "because it reduces variance through segment averaging while retaining "
                "O(N log N) FFT efficiency."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_partial_directed_coherence",
            mathematical_basis="Partial Directed Coherence (PDC) derived from VAR "
                               "coefficient matrix A(f) = I - Σ A_k e^{-i2πfk}; "
                               "vectorised over frequency bins",
            role="Computes frequency-resolved, directional coupling strength "
                 "(PDC) between attention heads from fitted VAR coefficients.",
            justification=(
                "PDC extends classical spectral analysis to the *multivariate* case: "
                "it reveals not only which frequencies carry power, but which heads "
                "drive which other heads at each frequency band.  This is essential for "
                "mapping the causal information flow inside the transformer at specific "
                "temporal scales (low-frequency = slow context drift; high-frequency = "
                "rapid token-level interactions)."
            ),
        ),
    ],

    # ── 2. Statistical Features ────────────────────────────────────────── #
    "statistical_features": [
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="stationarity_test",
            mathematical_basis="Augmented Dickey-Fuller (ADF) unit-root test for T≥30; "
                               "variance ratio (Var(Δx)/Var(x)) for T<30",
            role="Tests whether the primary SVD component of the hidden-state "
                 "trajectory is stationary or drifting.",
            justification=(
                "Many time-series models (AR, VAR, spectral inference) assume weak "
                "stationarity.  For LLM activations, non-stationarity (drifting mean or "
                "variance) indicates that the model is *actively updating its internal "
                "belief* across tokens — a key explainability signal.  Stationary "
                "trajectories suggest rote retrieval; non-stationary ones suggest "
                "genuine multi-step reasoning."
            ),
        ),
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="decompose",
            mathematical_basis="Classical additive decomposition Y_t = T_t + S_t + R_t "
                               "via moving-average trend and periodic seasonal extraction",
            role="Splits the primary SVD component into trend, seasonal, and "
                 "residual sub-signals.",
            justification=(
                "Decomposing the residual stream into trend (long-range belief drift), "
                "seasonality (periodic syntactic/semantic cycles), and residual (noise "
                "/ unexplained variance) directly supports the project objective of "
                "understanding *what* the LLM is computing at each token position.  "
                "A rising trend indicates semantic commitment; strong seasonality "
                "indicates repetitive reasoning patterns."
            ),
        ),
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="decompose_feature_space",
            mathematical_basis="First-difference energy, rolling mean/variance, "
                               "z-scored regime energy score",
            role="Decomposes multivariate attention-feature series "
                 "(T × H × F) into trend, seasonal, residual, rolling volatility, "
                 "and a scalar regime-change score per token.",
            justification=(
                "Attention features are higher-dimensional than the residual stream "
                "and require a more robust pipeline.  First-difference energy "
                "Σ|Δx_t| is a model-free measure of how quickly the hidden dynamics "
                "are changing, making it a useful state fingerprint for comparing "
                "token windows without any distributional assumptions."
            ),
        ),
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="calculate_trajectory_dynamics",
            mathematical_basis="L2-norm velocity ||h_t − h_{t-1}||₂, "
                               "cosine-similarity momentum, acceleration Δ‖v‖, "
                               "and Hurst exponent via Detrended Fluctuation Analysis "
                               "(DFA log-log regression)",
            role="Computes step-by-step velocity, momentum, acceleration, and "
                 "the DFA-based Hurst exponent of the hidden-state trajectory.",
            justification=(
                "Velocity spikes identify semantic transition points (topic changes, "
                "reasoning phase shifts).  The Hurst exponent H characterises the "
                "long-range dependence of the activation signal: H>0.5 indicates "
                "persistent (logically coherent) reasoning; H<0.5 indicates "
                "anti-persistent (random-walk / hallucination-prone) dynamics.  "
                "These are direct, interpretable features for explaining *how* the "
                "model progresses through its answer."
            ),
        ),
    ],

    # ── 3. Autocorrelation / Lag Analysis ─────────────────────────────── #
    "autocorrelation_lag": [
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="autocorrelation_analysis",
            mathematical_basis="ACF and PACF via statsmodels (FFT-accelerated); "
                               "95 % confidence band 2/√N",
            role="Computes the autocorrelation (ACF) and partial autocorrelation "
                 "(PACF) functions of the primary SVD component.",
            justification=(
                "ACF/PACF directly maps to the look-back mechanism of transformer "
                "attention: significant lag-k ACF means the hidden state at token t "
                "is correlated with the state at token t-k, reflecting attention "
                "span.  PACF reveals the *direct* order of temporal dependence after "
                "removing shorter-lag contributions, informing AR/VAR lag selection."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_select_optimal_lag",
            mathematical_basis="AIC-minimising VAR lag selection via "
                               "statsmodels VAR.select_order()",
            role="Chooses the optimal VAR lag order p that minimises the Akaike "
                 "Information Criterion for a multivariate head series.",
            justification=(
                "Choosing an appropriate lag p prevents both under-fitting (missing "
                "real temporal dependencies) and over-fitting (spurious dependencies "
                "from too many parameters).  AIC-based selection provides a "
                "principled, data-driven answer grounded in information theory."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_correlation_fallback",
            mathematical_basis="Time-lagged Pearson correlation matrix at lag ℓ; "
                               "Corr(X_i(t), X_j(t−ℓ))",
            role="Computes a time-lagged pairwise correlation matrix as a "
                 "robust, model-free fallback when VAR/VECM cannot be fitted "
                 "on short samples.",
            justification=(
                "Real-time analysis on short token windows (T<30) cannot reliably "
                "support full VAR identification.  The lagged Pearson matrix still "
                "captures directional predictive influence between heads at a chosen "
                "lag, providing an interpretable approximation of inter-head temporal "
                "coupling without requiring large sample sizes."
            ),
        ),
    ],

    # ── 4. Dimensionality Reduction ────────────────────────────────────── #
    "dimensionality_reduction": [
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="svd_compress",
            mathematical_basis="Economy SVD: X = UΣVᵀ → compressed = U[:,:k]·Σ[:k]; "
                               "optional Hankel (delay-embedding) augmentation",
            role="Reduces the [Tokens × HiddenDim] residual stream to "
                 "[Tokens × n_components] via truncated SVD.",
            justification=(
                "LLM hidden states typically live in ℝ^{4096} or larger.  Direct "
                "time-series analysis at full dimension is computationally intractable "
                "and statistically noisy.  SVD extracts the dominant orthogonal "
                "directions of variance — the *principal dynamic modes* — in a "
                "mathematically optimal (Eckart-Young theorem) sense, making "
                "downstream spectral, ACF, and trajectory analysis both tractable and "
                "interpretable.  The optional delay embedding builds a Hankel-like "
                "matrix so that SVD captures temporal correlations, not just "
                "instantaneous covariance."
            ),
        ),
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="incremental_svd_update",
            mathematical_basis="Sketch-based incremental SVD (randomised_svd on new "
                               "rows only) + sign-alignment; Brand's rank-1 update "
                               "approximation",
            role="Extends an existing SVD projection when the trajectory grows "
                 "beyond the cache size, without re-processing all past tokens.",
            justification=(
                "For long generation sequences the O((T_old+T_new)×D) cost of "
                "full re-SVD becomes prohibitive.  The sketch-based update processes "
                "only new rows at O(T_new × n²) cost while preserving the quality "
                "of the compressed representation, enabling real-time streaming "
                "analysis consistent with the project's on-device deployment goal."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_extract_global_pca_factors",
            mathematical_basis="PCA via sklearn decomposition (SVD under the hood); "
                               "z-score normalisation of factors",
            role="Extracts top-k PCA factors from the hidden states to serve as "
                 "exogenous global drift regressors in a FAVAR model.",
            justification=(
                "Attention head time series can be spuriously correlated due to "
                "a shared global context drift (e.g. the model's overall confidence "
                "level evolving across the response).  PCA factors capture this "
                "global variance so it can be statistically controlled for in VAR "
                "causality tests, preventing false positive inter-head influence "
                "attributions."
            ),
        ),
    ],

    # ── 5. Change-point / State Detection ─────────────────────────────── #
    "changepoint_state_detection": [
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="compute_intrinsic_time",
            mathematical_basis="Arc-length parameterisation τ = Σ‖h_t − h_{t-1}‖₂; "
                               "discrete curvature κ_t = ‖Δ²h_t‖ / ‖Δh_t‖²; "
                               "change-points at κ_t > μ_κ + 2σ_κ",
            role="Computes cumulative arc-length (intrinsic time) and discrete "
                 "trajectory curvature, flagging high-curvature tokens as "
                 "geometric change-points in the hidden-state manifold.",
            justification=(
                "A sudden change in the direction of the hidden-state trajectory "
                "in activation space signals a semantic phase transition (e.g. a "
                "reasoning pivot, belief update, or topic shift).  Curvature-based "
                "change-point detection is purely geometric, requires no labelled "
                "data, and is robust to the non-stationarity of LLM activations, "
                "making it a principled complement to statistical change-point tests."
            ),
        ),
        FunctionEntry(
            module="chronoscope.observer",
            class_name="SignalObserver",
            function="incremental_analysis",
            mathematical_basis="Euler characteristic χ = V − E on thresholded distance "
                               "graph; anomaly if |Δχ| ≥ 2 or variance ratio > 1.5",
            role="Tracks the Euler characteristic of the local hidden-state "
                 "neighbourhood graph as a topological proxy for structural "
                 "phase transitions in the trajectory.",
            justification=(
                "The Euler characteristic is a computationally cheap topological "
                "invariant (O(window²)) that summarises the connectivity of the "
                "hidden-state manifold.  A sudden change in χ indicates that the "
                "geometric structure of the local activation neighbourhood has "
                "reorganised — a hallmark of reasoning phase boundaries such as "
                "prompt boundaries, tool calls, or belief updates."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_discover_phases_hmm",
            mathematical_basis="Gaussian HMM (hmmlearn) with full covariance; "
                               "persistence bias on transition matrix; "
                               "model selection via AIC/BIC",
            role="Fits a Gaussian Hidden Markov Model to the attention head "
                 "dynamics to infer discrete reasoning phases (states) and "
                 "their transition probabilities.",
            justification=(
                "An HMM models the LLM's internal activity as a latent-state "
                "process: the hidden state at each token is a discrete reasoning "
                "mode (e.g. context-gathering, inference, output-generation).  "
                "State sequences, emission means, and transition matrices are "
                "directly interpretable for explaining *what stage* of processing "
                "the model is in at each token position."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_segment_by_topological_phases",
            mathematical_basis="Threshold-based spike detection on Euler characteristic "
                               "series: threshold = μ_χ + σ_χ · spike_threshold_std",
            role="Converts the rolling Euler characteristic time series into "
                 "discrete token-span phase segments using spike boundaries.",
            justification=(
                "Segmenting the generation trace into topologically homogeneous "
                "phases enables per-phase aggregation of attention metrics (entropy, "
                "coupling strength), making it possible to compare the model's "
                "behaviour during distinct reasoning stages across different prompts "
                "or model variants."
            ),
        ),
    ],

    # ── 6. Evaluation / Validation Utilities ──────────────────────────── #
    "evaluation_validation": [
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_granger_pvalue_matrix",
            mathematical_basis="Granger causality F-test via statsmodels "
                               "VARResults.test_causality(); pval_matrix[i,j] = p(j→i)",
            role="Computes the full [H × H] matrix of Granger-causality F-test "
                 "p-values for every directed pair of attention heads.",
            justification=(
                "Granger causality provides a statistically testable (though "
                "predictive, not interventional) measure of whether past values of "
                "head j improve the forecast of head i above and beyond head i's own "
                "history.  The p-value matrix quantifies inter-head influence in a "
                "way that can be reported with confidence bounds, directly supporting "
                "the project's goal of *mathematically verifiable* reasoning analysis."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_apply_fdr_correction",
            mathematical_basis="Benjamini-Hochberg FDR correction via "
                               "statsmodels multipletests(method='fdr_bh')",
            role="Applies FDR correction to the Granger p-value matrix to "
                 "control the false-discovery rate across multiple simultaneous "
                 "hypothesis tests.",
            justification=(
                "Testing H(H-1) directional pairs simultaneously inflates the "
                "family-wise type-I error.  Benjamini-Hochberg correction keeps the "
                "expected proportion of false discoveries at a controlled level α, "
                "ensuring that reported inter-head connections survive rigorous "
                "statistical scrutiny rather than arising from multiple-comparison "
                "artefacts."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_bootstrap_surrogate_pvalues",
            mathematical_basis="Phase-scrambling (FFT magnitude-preserving surrogate "
                               "generation) + empirical null distribution of VAR L1 "
                               "influence norms over n_surrogates resamples",
            role="Derives empirical p-values for VAR influence scores by "
                 "comparing observed scores against a phase-scrambled null "
                 "distribution (n_surrogates = 500 by default).",
            justification=(
                "Parametric Granger tests assume Gaussian residuals and correct VAR "
                "specification.  Surrogate-based empirical p-values are "
                "distribution-free and robust to misspecification, providing a "
                "second, independent validation path for inter-head influence claims."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_conditional_transfer_entropy",
            mathematical_basis="Conditional transfer entropy TE(j→i) = "
                               "I(X_i(t+1); X_j^{1..k} | X_i^{1..k}) estimated via "
                               "discrete joint histograms",
            role="Estimates the non-linear directional information flow from "
                 "head j to head i conditioned on head i's own history.",
            justification=(
                "Transfer entropy is a model-free, information-theoretic measure "
                "that captures both linear and non-linear dependencies.  Unlike "
                "Granger causality, it requires no VAR model assumption, making it a "
                "complementary validation tool: agreement between Granger p-values "
                "and TE scores increases confidence that a detected inter-head "
                "influence is real."
            ),
        ),
        FunctionEntry(
            module="chronoscope.analyzer",
            class_name="CausalAnalyzer",
            function="_calculate_var_stability",
            mathematical_basis="Companion matrix eigenvalue test: VAR(p) is stable iff "
                               "all eigenvalues of the (H·p × H·p) companion matrix "
                               "lie inside the unit circle",
            role="Validates that a fitted VAR model is dynamically stable "
                 "(all companion-matrix eigenvalues have modulus < 1).",
            justification=(
                "An unstable VAR would have explosive, divergent forecast paths, "
                "which would invalidate all downstream Granger, PDC, and spectral "
                "causality inferences.  Stability checking is therefore a prerequisite "
                "quality gate before any causal claims are reported."
            ),
        ),
    ],
}


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------

def list_categories() -> List[str]:
    """Return the names of all tool categories in the mapping."""
    return list(MATH_TOOLS_MAP.keys())


def get_mapping() -> Dict[str, List[FunctionEntry]]:
    """Return the complete mapping dict (category → list of FunctionEntry)."""
    return MATH_TOOLS_MAP


def get_category(name: str) -> List[FunctionEntry]:
    """
    Return all FunctionEntry objects for a given category name.

    Parameters
    ----------
    name : str
        One of the keys returned by :func:`list_categories`.

    Raises
    ------
    KeyError
        If *name* is not a recognised category.
    """
    if name not in MATH_TOOLS_MAP:
        raise KeyError(
            f"Unknown category {name!r}. "
            f"Available categories: {list_categories()}"
        )
    return MATH_TOOLS_MAP[name]


def describe(category: Optional[str] = None) -> str:
    """
    Build a human-readable report of the code-grounded mapping.

    Parameters
    ----------
    category : str, optional
        If given, only entries for that category are included.

    Returns
    -------
    str
        Formatted text report suitable for printing or writing to a file.
    """
    lines: List[str] = [
        "=" * 72,
        "CHRONOSCOPE — Mathematical Tools: Code-Grounded Mapping",
        "Objective: Model LLM Internal Activities as Time Series",
        "=" * 72,
    ]

    categories = {category: MATH_TOOLS_MAP[category]} if category else MATH_TOOLS_MAP

    for cat_key, entries in categories.items():
        category_title = cat_key.replace("_", " ").title()
        lines += ["", f"▸ {category_title}", "─" * 64]

        for entry in entries:
            lines += [
                f"  Function : {entry.qualified_name}",
                f"  Math     : {entry.mathematical_basis}",
                f"  Role     : {entry.role}",
                f"  Why      : {entry.justification}",
                "",
            ]

    return "\n".join(lines)
