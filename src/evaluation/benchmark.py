
from src.inference.generation import generate_text
from src.evaluation.metrics import memory_metrics


def benchmark(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    device,
    precision,
    model_key,
    model_name,
    **extra,
):
    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    result.update(memory_metrics())

    result.update(
        {
            "precision": precision,
            "model_key": model_key,
            "model": model_name,
            "memory_measurement_scope": "generation_phase",
        }
    )

    result.update(extra)

    return result
