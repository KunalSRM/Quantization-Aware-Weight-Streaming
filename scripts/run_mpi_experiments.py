import argparse
import json
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
    get_project_root,
    load_config,
)

from src.distributed.mpi_runner import (
    MPIRunner,
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
            "MPI experiment "
            "orchestration for "
            "independent QAWS jobs."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(registry),
        default=[
            cfg["model"]["default"]
        ],
    )

    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=[
            "FP16",
            "INT8",
            "INT4",
            "INT3",
            "INT2",
        ],
        default=[
            "FP16",
            "INT4",
        ],
    )

    args = parser.parse_args()

    mpi = MPIRunner()

    jobs = [
        (model, precision)
        for model in args.models
        for precision in args.precisions
    ]

    local_jobs = mpi.distribute(
        jobs
    )

    print(
        f"MPI rank "
        f"{mpi.rank}/{mpi.size}: "
        f"{len(local_jobs)} local jobs"
    )

    local_results = []

    for (
        model_key,
        precision,
    ) in local_jobs:

        try:

            result = run_experiment(
                model_key,
                precision,
            )

            result["mpi_rank"] = (
                mpi.rank
            )

            local_results.append(
                result
            )

        except Exception as exc:

            local_results.append(
                {
                    "model_key": model_key,
                    "precision": precision,
                    "status": "failed",
                    "error": str(exc),
                    "mpi_rank": mpi.rank,
                }
            )

    gathered = mpi.gather(
        local_results
    )

    if mpi.rank == 0:

        flattened = [
            item
            for group in gathered
            for item in group
        ]

        output = (
            get_project_root()
            / "results"
            / "benchmarks"
            / "mpi_latest_results.json"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                flattened,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"Saved: {output}"
        )


if __name__ == "__main__":
    main()