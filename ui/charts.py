
import plotly.express as px


def _chart_data(df, metric):
    if df.empty or metric not in df.columns:
        return df.iloc[0:0]

    result = df.copy()
    result[metric] = __import__("pandas").to_numeric(
        result[metric],
        errors="coerce",
    )
    return result.dropna(subset=[metric])


def throughput_chart(df):
    data = _chart_data(df, "tokens_per_second")
    return px.bar(
        data,
        x="label",
        y="tokens_per_second",
        color="precision" if "precision" in data.columns else None,
        title="Generation Throughput",
        labels={
            "label": "Model / Precision",
            "tokens_per_second": "Tokens per second",
        },
    )


def latency_chart(df):
    data = _chart_data(df, "latency_s")
    return px.bar(
        data,
        x="label",
        y="latency_s",
        color="precision" if "precision" in data.columns else None,
        title="End-to-End Generation Latency",
        labels={
            "label": "Model / Precision",
            "latency_s": "Latency (seconds)",
        },
    )


def memory_chart(df):
    data = _chart_data(df, "gpu_peak_gb")
    return px.bar(
        data,
        x="label",
        y="gpu_peak_gb",
        color="precision" if "precision" in data.columns else None,
        title="Peak GPU Memory",
        labels={
            "label": "Model / Precision",
            "gpu_peak_gb": "Peak GPU memory (GB)",
        },
    )
