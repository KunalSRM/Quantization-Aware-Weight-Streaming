import gc
import json

from datetime import (
    datetime,
    timezone,
)

import torch

from src.core.config import (
    get_model_config,
    get_project_root,
    load_config,
)

from src.models.model_loader import (
    get_device,
    load_model,
)

from src.quantization.sensitivity import (
    analyze_tensor,
)

from src.streaming.apply import (
    apply_streaming,
)

from src.evaluation.benchmark import (
    benchmark,
)


PRECISION_BITS = {
    "INT8": 8,
    "INT4": 4,
    "INT3": 3,
    "INT2": 2,
}


SUPPORTED_PRECISIONS = {
    "FP16",
    *PRECISION_BITS.keys(),
}


def save_benchmark_result(
    result,
):

    out = (
        get_project_root()
        / "results"
        / "benchmarks"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    safe_key = (
        result["model_key"]
        .replace("/", "_")
    )

    path = (
        out
        / f"{safe_key}_{result['precision']}_{stamp}.json"
    )

    path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out / "latest_experiment.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def _cleanup(
    model=None,
    tokenizer=None,
):

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def run_experiment(
    model_key,
    precision="FP16",
    config_path=None,
    save_result=True,
):

    cfg = load_config(
        config_path
    )

    model_cfg = get_model_config(
        cfg,
        model_key,
    )

    precision = precision.upper()

    if precision not in SUPPORTED_PRECISIONS:

        raise ValueError(
            f"Unsupported precision '{precision}'. "
            f"Supported: "
            f"{', '.join(sorted(SUPPORTED_PRECISIONS))}"
        )

    if (
        precision != "FP16"
        and not cfg["streaming"].get(
            "enabled",
            True,
        )
    ):

        raise ValueError(
            "INT experiments require "
            "streaming.enabled=true"
        )

    device = get_device(
        cfg["hardware"]["device"],
        cfg["hardware"].get(
            "allow_cpu_fallback",
            False,
        ),
    )

    tokenizer = None
    model = None

    try:

        tokenizer, model = load_model(
            model_cfg["name"],
            device,
        )

        if device.type == "cuda":

            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats()

        common = {
            "family": model_cfg["family"],
            "size_b": model_cfg["size_b"],
            "tier": model_cfg["tier"],
        }

        prompt = cfg[
            "benchmark"
        ]["prompt"]

        max_new_tokens = int(
            cfg[
                "benchmark"
            ]["max_new_tokens"]
        )

        # =================================
        # FP16 BASELINE
        # =================================

        if precision == "FP16":

            result = benchmark(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                precision=precision,
                model_key=model_key,
                model_name=model_cfg["name"],
                streaming_enabled=False,
                **common,
            )

        # =================================
        # QUANTIZED + STREAMING
        # =================================

        else:

            bits = PRECISION_BITS[
                precision
            ]

            sensitivity_all = {}

            target = getattr(
                model,
                "lm_head",
                None,
            )

            if (
                target is not None
                and hasattr(
                    target,
                    "weight",
                )
            ):

                sensitivity_all = (
                    analyze_tensor(
                        target.weight,
                        tuple(
                            PRECISION_BITS.values()
                        ),
                    )
                )

            streamed_layers = (
                apply_streaming(
                    model,
                    bits,
                    device,
                )
            )

            result = benchmark(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                precision=precision,
                model_key=model_key,
                model_name=model_cfg["name"],
                streaming_enabled=True,
                streamed_linear_layers=streamed_layers,
                sensitivity=sensitivity_all.get(
                    precision
                ),
                sensitivity_all=sensitivity_all,
                sensitivity_target="lm_head.weight",
                **common,
            )

        result[
            "timestamp_utc"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        result["status"] = "success"

        if save_result:

            result[
                "result_file"
            ] = str(
                save_benchmark_result(
                    result
                )
            )

        return result

    finally:

        _cleanup(
            model,
            tokenizer,
        )