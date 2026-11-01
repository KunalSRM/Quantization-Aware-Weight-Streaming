import argparse
import sys

from pathlib import Path


sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    ),
)


from src.core.config import (
    load_config,
)

from src.pipeline.experiment import (
    run_experiment,
)


def main():

    cfg = load_config()

    model_keys = list(
        cfg["model"]["registry"]
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run one QAWS "
            "model/precision experiment."
        )
    )

    parser.add_argument(
        "--model",
        choices=model_keys,
        default=cfg[
            "model"
        ]["default"],
    )

    parser.add_argument(
        "--precision",
        choices=[
            "FP16",
            "INT8",
            "INT4",
            "INT3",
            "INT2",
        ],
        default="FP16",
    )

    args = parser.parse_args()

    result = run_experiment(
        args.model,
        args.precision,
    )

    print(
        "\n===== Experiment Result ====="
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()