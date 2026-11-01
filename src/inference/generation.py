import json
import os
import time
from pathlib import Path

import torch

from src.core.exceptions import InferenceError


def _system_ram_gb():
    """Return system RAM usage in GB when psutil is available."""
    try:
        import psutil

        return psutil.virtual_memory().used / (1024 ** 3)
    except ImportError:
        return None
    except Exception:
        return None


def _gpu_metrics(device):
    """Collect actual CUDA memory metrics."""
    metrics = {
        "gpu_allocated_gb": None,
        "gpu_reserved_gb": None,
        "gpu_peak_gb": None,
    }

    try:
        if device is not None and device.type == "cuda":
            metrics.update({
                "gpu_allocated_gb": (
                    torch.cuda.memory_allocated(device) / (1024 ** 3)
                ),
                "gpu_reserved_gb": (
                    torch.cuda.memory_reserved(device) / (1024 ** 3)
                ),
                "gpu_peak_gb": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                ),
            })
    except Exception:
        pass

    return metrics


class _TelemetryWriter:
    """Append timestamped inference measurements to a JSONL file."""

    def __init__(self, path):
        self.path = path
        self.enabled = bool(path)

        if self.enabled:
            try:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
            except OSError as exc:
                raise InferenceError(
                    f"Cannot initialize telemetry file: {exc}"
                ) from exc

    def write(self, data):
        if not self.enabled:
            return

        try:
            with open(self.path, "a", encoding="utf-8") as file:
                file.write(json.dumps(data, default=str) + "\n")
                file.flush()
        except OSError:
            # Telemetry failures should not interrupt model inference.
            self.enabled = False


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=64,
    device=None,
):
    device = device or next(model.parameters()).device

    try:
        if (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template
        ):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        input_tokens = int(inputs["input_ids"].shape[-1])
        telemetry = _TelemetryWriter(
            os.environ.get("QAWS_TELEMETRY_PATH")
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        last_sample = started
        last_tokens = 0

        def record_progress(token_count, force=False):
            nonlocal last_sample, last_tokens

            now = time.perf_counter()
            interval = now - last_sample
            elapsed = now - started

            # Sampling every 0.5 seconds avoids excessive disk writes,
            # while still capturing every generated token count.
            if not force and interval < 0.5:
                return

            delta_tokens = token_count - last_tokens

            telemetry.write({
                "event": "generation",
                "elapsed_s": round(elapsed, 4),
                "generated_tokens": token_count,
                "target_tokens": int(max_new_tokens),
                "progress": min(
                    token_count / max_new_tokens, 1.0
                ) if max_new_tokens > 0 else 1.0,
                "tokens_per_second": (
                    delta_tokens / interval if interval > 0 else 0.0
                ),
                "average_tokens_per_second": (
                    token_count / elapsed if elapsed > 0 else 0.0
                ),
                "ram_gb": _system_ram_gb(),
                **_gpu_metrics(device),
            })

            last_sample = now
            last_tokens = token_count

        # Hugging Face calls stopping criteria after decoding steps.
        # Returning False lets generation continue normally.
        from transformers import StoppingCriteria, StoppingCriteriaList

        class TelemetryCriteria(StoppingCriteria):
            def __call__(
                self,
                input_ids,
                scores,
                **kwargs,
            ):
                generated = max(
                    0, int(input_ids.shape[-1]) - input_tokens
                )
                record_progress(generated)
                return False

        telemetry.write({
            "event": "started",
            "elapsed_s": 0.0,
            "generated_tokens": 0,
            "target_tokens": int(max_new_tokens),
            "progress": 0.0,
            "ram_gb": _system_ram_gb(),
            **_gpu_metrics(device),
        })

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([
                    TelemetryCriteria()
                ]),
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - started
        total_tokens = int(output.shape[-1])
        generated_tokens = max(0, total_tokens - input_tokens)

        # Always record the final sample, even for short generations.
        record_progress(generated_tokens, force=True)

        text_out = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        final_metrics = {
            "event": "completed",
            "elapsed_s": latency,
            "generated_tokens": generated_tokens,
            "target_tokens": int(max_new_tokens),
            "progress": 1.0 if max_new_tokens > 0 else 1.0,
            "tokens_per_second": (
                generated_tokens / latency if latency > 0 else 0.0
            ),
            "average_tokens_per_second": (
                generated_tokens / latency if latency > 0 else 0.0
            ),
            "ram_gb": _system_ram_gb(),
            **_gpu_metrics(device),
        }
        telemetry.write(final_metrics)

        return {
            "text": text_out,
            "latency_s": latency,
            "input_tokens": input_tokens,
            "total_tokens": total_tokens,
            "generated_tokens": generated_tokens,
            "tokens_per_second": (
                generated_tokens / latency if latency > 0 else 0.0
            ),
        }

    except Exception as exc:
        if isinstance(exc, InferenceError):
            raise
        raise InferenceError(str(exc)) from exc
