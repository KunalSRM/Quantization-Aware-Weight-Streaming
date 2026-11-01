import argparse
import gc
import json
import sys
import time

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    ),
)


import torch


from src.core.config import (
    get_project_root,
    load_config,
)

from src.pipeline.experiment import (
    run_experiment,
)


def main():

    cfg = load_config()

    registry = cfg[
        "model"
    ]["registry"]

    parser = argparse.ArgumentParser(
        description=(
            "Sequential multi-model "
            "QAWS sweep. "
            "Models are run one at a time."
        )
    )

    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(registry),
    )

    parser.add_argument(
        "--precisions",
        nargs="+",
        default=[
            "FP16",
            "INT4",
        ],
        choices=[
            "FP16",
            "INT8",
            "INT4",
            "INT3",
            "INT2",
        ],
    )

    parser.add_argument(
        "--safe-only",
        action="store_true",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    keys = (
        args.models
        or list(registry)
    )

    if args.safe_only:

        keys = [
            k
            for k in keys
            if registry[k]["tier"]
            == "safe"
        ]

    if not keys:

        raise SystemExit(
            "No models selected."
        )

    results = []

    started = datetime.now(
        timezone.utc
    ).isoformat()

    print(
        "\nSelected models:"
    )

    for key in keys:

        info = registry[key]

        print(
            f"  - {key}: "
            f"{info['family']} | "
            f"{info['size_b']}B | "
            f"{info['tier']}"
        )

    print(
        f"Precisions: "
        f"{', '.join(args.precisions)}\n"
    )

    for model_key in keys:

        for precision in args.precisions:

            print(
                f"\n===== "
                f"{model_key} | "
                f"{precision} ====="
            )

            try:

                result = run_experiment(
                    model_key,
                    precision,
                )

                results.append(
                    result
                )

                print(
                    f"SUCCESS | "
                    f"latency="
                    f"{result.get('latency_s', 0):.3f}s | "
                    f"throughput="
                    f"{result.get('tokens_per_second', 0):.3f} tok/s | "
                    f"peak_gpu="
                    f"{result.get('gpu_peak_gb', 0):.3f} GB"
                )

            except Exception as exc:

                failure = {
                    "model_key": model_key,
                    "family": registry[
                        model_key
                    ]["family"],
                    "size_b": registry[
                        model_key
                    ]["size_b"],
                    "precision": precision,
                    "status": "failed",
                    "error": str(exc),
                    "timestamp_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                }

                results.append(
                    failure
                )

                print(
                    f"FAILED: {exc}"
                )

            finally:

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

                time.sleep(
                    max(
                        0.0,
                        args.delay,
                    )
                )

    output = {

        "started_utc": started,

        "finished_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "models": keys,

        "precisions": (
            args.precisions
        ),

        "results": results,
    }

    out_dir = (
        get_project_root()
        / "results"
        / "benchmarks"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file = (
        out_dir
        / "model_sweep_latest.json"
    )

    out_file.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved sweep summary: "
        f"{out_file}"
    )


if __name__ == "__main__":
    main()