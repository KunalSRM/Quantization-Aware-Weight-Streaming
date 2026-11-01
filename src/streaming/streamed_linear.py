
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantization.quantizer import (
    UniformQuantizer,
    QuantizedTensor,
    dequantize_tensor,
)


class StreamedLinear(nn.Module):
    """Store packed quantized weights on CPU and dequantize for forward."""

    def __init__(self, linear: nn.Linear, bits: int) -> None:
        super().__init__()

        quantizer = UniformQuantizer(bits)
        quantized = quantizer.quantize(linear.weight.detach())

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.original_shape = tuple(linear.weight.shape)
        self.numel = linear.weight.numel()
        self.qmin = quantized.qmin

        # Registered buffers are included in state_dict().
        self.register_buffer(
            "packed_weight",
            quantized.data.detach().cpu().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "scale",
            quantized.scale.detach().cpu().clone(),
            persistent=True,
        )

        if linear.bias is not None:
            self.register_buffer(
                "bias",
                linear.bias.detach().cpu().clone().contiguous(),
                persistent=True,
            )
        else:
            self.register_buffer("bias", None, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized = QuantizedTensor(
            data=self.packed_weight,
            scale=self.scale,
            bits=self.bits,
            original_shape=self.original_shape,
            numel=self.numel,
            qmin=self.qmin,
        )

        weight = dequantize_tensor(
            quantized,
            device=x.device,
            dtype=x.dtype,
        )

        bias = (
            self.bias.to(device=x.device, dtype=x.dtype)
            if self.bias is not None
            else None
        )

        return F.linear(x, weight, bias)
