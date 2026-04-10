"""
Chronoscope: A Glass-Box Observability Engine for LLM Reasoning Traces.

Treats LLM hidden states as multivariate time series and applies classical
signal processing + causal inference to mathematically verify reasoning validity.
"""

__version__ = "0.1.0"

from .config import ChronoscopeConfig
from .models import load_model, list_hookable_layers, detect_num_attention_heads, detect_hidden_dim
from .interceptor import ChronoscopeInterceptor
from .observer import SignalObserver
from .analyzer import CausalAnalyzer
from .synthesizer import ReportSynthesizer
from .dashboard_bridge import DashboardBridge
from .math_tools_map import (
    MATH_TOOLS_MAP,
    FunctionEntry,
    list_categories,
    get_mapping,
    get_category,
    describe,
)
