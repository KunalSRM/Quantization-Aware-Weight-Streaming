from src.pipeline.experiment import (
    run_experiment,
)


def run_worker(
    model_key,
    precision,
    rank=0,
):

    result = run_experiment(
        model_key,
        precision,
    )

    result["mpi_rank"] = rank

    return result