# Chronoscope: A Spectral-Topological Field Theory of LLM Reasoning
### Technical Whitepaper & Mathematical Specification

## 1. Abstract
Chronoscope treats the high-dimensional hidden state residual stream of Large Language Models (LLMs) as a non-stationary vector stochastic process. By applying rigorous spectral, topological, and causal-time-series tools, we quantify the internal "Coherence" and "Fidelity" of the model's reasoning trajectory. This framework enables real-time detection of logical persistence, cognitive hesitation, and structural breaks (hallucinations).

---

## 2. The Core Mathematical Pillars

The system decomposes the trajectory into a hierarchical set of "Observable" dynamics.

| Mathematical Tool | Formal Specification | Use in Chronoscope | Rationale (Selection Basis) |
| :--- | :--- | :--- | :--- |
| **SVD (Rank-k)** | $\mathbf{X} = \mathbf{U \Sigma V}^T$ | Transforms $D=896+$ hidden dims into $k=16$ principal axes of logic. | **Determinism & Interpretability**. Unlike t-SNE, SVD is non-stochastic and allows an inverse mapping back to specific neurons for interpretability. |
| **TDA (Topology)** | $VR(X, \epsilon)$ (Vietoris-Rips) | Calculates **Betti-1** ($\beta_1$) loops and **Euler Characteristic** ($\chi$). | **Structural Invariance**. Captures "loops" where the model revisits premises. It detects the *shape* of reasoning, not just the magnitude of activations. |
| **VECM (Cointegration)**| $\Delta x_t = \Pi x_{t-1} + \sum \Gamma_i \Delta x_{t-i}$ | Models multiple attention heads converging toward a single logical output. | **Handling Belief Drift**. Reasoning is non-stationary ($I(1)$); VECM models both short-term "shocks" and long-term "consensus" paths. |
| **DFA (Hurst)** | $F(n) \propto n^H$ | Measures the **Roughness** and "Memory" (long-range dependence) of the stream. | **Non-Stationary Logic**. Standard $R/S$ fails on trending signals; DFA extracts coherence from the "Reasoning Drift" without being fooled by it. |
| **Granger (Wald)** | $H_0: \Phi_{ij} = [0]$ | Populates a **Directed Influence Matrix** of head-to-head dynamics. | **Causality vs. Correlation**. Attention weights show who is *viewed*; Granger proves whose hidden state *determines* another's future state. |
| **DTW (Warping)** | $\min \sum d(w_i, w_j)$ | Compares diverse layers that process logic at different "Temporal Speeds." | **Temporal Elasticity**. Logic processes at different token-rates across the transformer; DTW aligns the *features*, not just the *timestamps*. |
| **JSD (Divergence)** | $JSD(P\|Q) \in [0, 1]$ | Aggregates per-head "Fidelity Scores" into a normalized **Reasoning Verdict**. | **Metric Symmetry**. Unlike KL-Divergence, JSD is symmetric and bounded, making it ideal for stable Z-score aggregation across 100+ heads. |

---

## 3. Managing Statistical Assumptions (The Bridge)

To apply classical time series mathematics to deterministic LLM hidden states, we utilize the following "Translation Layer":

*   **Handling Non-Stochasticity**: 
    *   *Assumption*: Time series tools assume random noise ($\epsilon_t$). LLMs are deterministic.
    *   *Real-World Solution*: We treat the hidden stream as a **State-Space Model** where "Innovation" ($\epsilon_t$) is not noise, but the **Information Gain** provided by each consecutive token being processed.
*   **Stationarity & Drift**: 
    *   *Assumption*: Regression requires stationary inputs. Reasoning paths are inherently non-stationary (they drift toward a conclusion).
    *   *Real-World Solution*: We apply the **Augmented Dickey-Fuller (ADF)** test as a gatekeeper. If reasoning is trending ($I(1)$), we utilize **First-Differencing** or **VECM** to reveal the underlying stable causality beneath the trend.
*   **The Distribution Gap**:
    *   *Assumption*: Many tests assume Gaussian (Normal) distributions. Activations are often sparse or fat-tailed.
    *   *Real-World Solution*: We utilize **Non-Parametric TDA** (which depends only on distance geometry) and **Robust Wald Tests** with **White Standard Errors** to protect against statistical artifacts.

### Formal Test Assumptions (VAR, ADF, KPSS)

*   **Vector Autoregression (VAR)**
    *   *Statistical Assumption*: Requires **covariance-stationarity** across all time series vectors. Residuals (innovations) must be multivariate white noise (zero mean, no serial correlation, constant variance). Assumes no perfect multicollinearity between endogenous variables.
    *   *Chronoscope Context*: Because raw attention head outputs exhibit high multicollinearity and drift, we apply SVD dimensionality reduction and first-differencing before VAR fitting to satisfy stability constraints.
*   **Augmented Dickey-Fuller (ADF) Test**
    *   *Statistical Assumption*: Assumes the underlying process is an autoregressive model of order $p$. Residuals in the test regression must be independent and homoskedastic. **Null Hypothesis**: The time series contains a unit root (is non-stationary).
    *   *Chronoscope Context*: Acts as a gatekeeper. If ADF fails to reject the null, it indicates the LLM reasoning path is "trending" (accumulating unchecked context) rather than stabilizing on an answer.
*   **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**
    *   *Statistical Assumption*: Assumes the time series can be decomposed into a deterministic trend, a random walk, and a stationary error term. **Null Hypothesis**: The series is stationary around a deterministic trend (trend-stationary) or fixed level. 
    *   *Chronoscope Context*: Used in conjunction with ADF for strict cross-validation. Since ADF has low power against near-unit-root processes, KPSS ensures we correctly distinguish a deterministic reasoning trajectory from a true internal random walk.

---

## 4. Translating Math into LLM Reasoning States

How these abstract metrics describe the "Thinking" process for your project presentation:

| Mathematical Condition | Translated LLM Reasoning State |
| :--- | :--- |
| **Betti-1 ($\beta_1 > 0$)** | **Recursive Verification**: The model is "Circling Back" to cross-reference a previous premise against its current work. |
| **Cointegration Rank ($r > 0$)** | **Core Belief Consensus**: Multiple attention heads have "locked on" to the same answer and are now just generating supporting tokens. |
| **Persistent Hurst ($H > 0.8$)** | **Logical Commitment**: The model has stabilized its belief state and is following a highly predictable, confident path. |
| **Hurst Decay ($H \to 0.5$)** | **Cognitive Hesitation**: The model is "jittering" between two probable answers; high probability of a "Structural Break" (hallucination). |
| **High SVD-Variance Axis** | **Primary Reasoning Dimension**: This is the principal "Logical Work" being done (e.g., Code syntax vs. Semantic intent). |
| **Granger Leading Head** | **The Orchestrator**: The head that is currently driving the rest of the model's computation toward the final token. |

---

## 5. Implementation Stack
*   **Trajectory Capture**: `ChronoscopeInterceptor` (PyTorch forward hooks)
*   **Spectral/Stats**: `SignalObserver` (`statsmodels`, `scipy.signal`)
*   **Topology**: `CausalAnalyzer` (`ripser`, `scipy.spatial`)
*   **Vector Dynamics**: `VAR` / `VECM` (`statsmodels.tsa.vector_ar`)
*   **Temporal Alignment**: `FastDTW`

---

> [!NOTE]
> This framework is designed for **High-Rigor Observability** in agentic AI. 
> The **Composite Fidelity Score** is the Z-score weighted average of these components:
> $$ \text{Fidelity} = \alpha(\text{TDA Persistence}) + \beta(\frac{1}{H}) + \gamma(\text{VAR Stability}) $$
