from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PATHS AND LOGGING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmarks"
LOG_DIR = ROOT / "results" / "logs"
TELEMETRY_DIR = ROOT / "results" / "telemetry"
EXPERIMENT_SCRIPT = ROOT / "scripts" / "run_experiment.py"

for directory in (RESULTS, LOG_DIR, TELEMETRY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qaws.dashboard")

PRECISIONS = ["FP16", "INT8", "INT4", "INT3", "INT2"]
DEFAULT_MODEL = "qwen_0_5b"

METRIC_COLUMNS = [
    "latency_s", "tokens_per_second", "gpu_peak_gb", "gpu_allocated_gb",
    "gpu_reserved_gb", "ram_gb", "input_tokens", "generated_tokens",
    "sensitivity", "size_b",
]
DISPLAY_COLUMNS = [
    "family", "model_key", "size_b", "precision", "latency_s", "input_tokens",
    "generated_tokens", "tokens_per_second", "gpu_allocated_gb",
    "gpu_reserved_gb", "gpu_peak_gb", "ram_gb", "streaming_enabled",
    "streamed_linear_layers", "sensitivity", "status", "mpi_rank", "source_file",
]
TELEMETRY_COLUMNS = [
    "elapsed_s", "generated_tokens", "target_tokens", "progress",
    "interval_tokens_per_second", "average_tokens_per_second", "ram_gb",
    "gpu_allocated_gb", "gpu_reserved_gb", "gpu_peak_gb",
]


# ============================================================
# OPTIONAL PLOTLY CHART HELPERS
# ============================================================

try:
    try:
        from charts import latency_chart, memory_chart, throughput_chart
    except ImportError:
        from ui.charts import latency_chart, memory_chart, throughput_chart
    CHART_IMPORT_ERROR = None
except Exception as exc:
    latency_chart = memory_chart = throughput_chart = None
    CHART_IMPORT_ERROR = str(exc)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Quantization-Aware Weight Streaming",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 12px 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Quantization-Aware Weight Streaming")
st.caption(
    "LLM inference experiments • precision comparison • live generation telemetry")