import torch

from src.quantization.quantizer import (
    UniformQuantizer,
)


def reconstruction_mse(
    tensor: torch.Tensor,
    bits: int,
) -> float:

    q = UniformQuantizer(bits)

    x = (
        tensor
        .detach()
        .float()
        .flatten()
        .cpu()
    )

    max_abs = x.abs().max()

    scale = (
        max_abs / q.qmax
        if max_abs > 0
        else torch.tensor(1.0)
    )

    qv = (
        torch.round(x / scale)
        .clamp(q.qmin, q.qmax)
    )

    recon = qv * scale

    return float(
        torch.mean(
            (x - recon) ** 2
        ).item()
    )


def analyze_tensor(
    tensor,
    bits_list=(8, 4, 3, 2),
):

    return {
        f"INT{b}": reconstruction_mse(
            tensor,
            b,
        )
        for b in bits_list
    }