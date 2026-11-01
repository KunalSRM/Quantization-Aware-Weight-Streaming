import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.core.exceptions import (
    DeviceError,
    ModelLoadError,
)

from src.core.logger import get_logger


log = get_logger(__name__)


def get_device(
    requested="cuda",
    allow_cpu_fallback=False,
):
    requested = requested.lower()

    if requested == "cuda":

        if torch.cuda.is_available():
            return torch.device("cuda")

        if allow_cpu_fallback:
            return torch.device("cpu")

        raise DeviceError(
            "CUDA was requested but is unavailable. "
            "CPU fallback is disabled."
        )

    return torch.device(requested)


def load_model(
    model_name: str,
    device: torch.device,
):
    try:

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {
            "low_cpu_mem_usage": True
        }

        if device.type == "cuda":
            kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = torch.float32

        try:

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **kwargs,
            )

        except TypeError:

            kwargs.pop("dtype", None)

            kwargs["torch_dtype"] = (
                torch.float16
                if device.type == "cuda"
                else torch.float32
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **kwargs,
            )

        model = model.to(device).eval()

        log.info(
            "Loaded %s on %s",
            model_name,
            device,
        )

        return tokenizer, model

    except Exception as e:

        raise ModelLoadError(
            f"Failed to load {model_name}: {e}"
        ) from e