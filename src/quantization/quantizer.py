from dataclasses import dataclass

import torch

from src.core.exceptions import QuantizationError


@dataclass
class QuantizedTensor:

    data: torch.Tensor
    scale: torch.Tensor
    bits: int
    original_shape: tuple
    numel: int
    qmin: int


class UniformQuantizer:

    def __init__(self, bits: int):

        if bits not in (2, 3, 4, 8):
            raise QuantizationError(
                "Supported bits: 2, 3, 4, 8"
            )

        self.bits = bits

        self.qmin = -(2 ** (bits - 1))

        self.qmax = (2 ** (bits - 1)) - 1

    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> QuantizedTensor:

        if not torch.is_floating_point(tensor):
            tensor = tensor.float()

        x = (
            tensor
            .detach()
            .float()
            .flatten()
            .cpu()
        )

        max_abs = x.abs().max()

        scale = (
            max_abs / self.qmax
            if max_abs > 0
            else torch.tensor(1.0)
        )

        q = (
            torch.round(x / scale)
            .clamp(self.qmin, self.qmax)
            .to(torch.int16)
        )

        if self.bits == 8:
            data = q.to(torch.int8)

        else:
            data = pack_bits(
                q,
                self.bits,
            )

        return QuantizedTensor(
            data=data,
            scale=scale.float(),
            bits=self.bits,
            original_shape=tuple(tensor.shape),
            numel=x.numel(),
            qmin=self.qmin,
        )


def pack_bits(
    q: torch.Tensor,
    bits: int,
) -> torch.Tensor:

    q = q.to(torch.int16).flatten()

    unsigned = (
        q - (-(2 ** (bits - 1)))
    ).to(torch.int16)

    # -------------------------
    # INT2
    # -------------------------

    if bits == 2:

        pad = (-len(unsigned)) % 4

        if pad:
            unsigned = torch.cat(
                [
                    unsigned,
                    torch.zeros(
                        pad,
                        dtype=torch.int16,
                    ),
                ]
            )

        v = unsigned.view(-1, 4).to(torch.uint8)

        return (
            v[:, 0]
            | (v[:, 1] << 2)
            | (v[:, 2] << 4)
            | (v[:, 3] << 6)
        ).contiguous()

    # -------------------------
    # INT4
    # -------------------------

    if bits == 4:

        pad = (-len(unsigned)) % 2

        if pad:
            unsigned = torch.cat(
                [
                    unsigned,
                    torch.zeros(
                        pad,
                        dtype=torch.int16,
                    ),
                ]
            )

        v = unsigned.view(-1, 2).to(torch.uint8)

        return (
            v[:, 0]
            | (v[:, 1] << 4)
        ).contiguous()

    # -------------------------
    # INT3
    # -------------------------

    if bits == 3:

        pad = (-len(unsigned)) % 8

        if pad:
            unsigned = torch.cat(
                [
                    unsigned,
                    torch.zeros(
                        pad,
                        dtype=torch.int16,
                    ),
                ]
            )

        v = unsigned.view(
            -1,
            8,
        ).to(torch.int32)

        out = torch.empty(
            v.shape[0],
            3,
            dtype=torch.uint8,
        )

        packed = sum(
            v[:, i] << (3 * i)
            for i in range(8)
        )

        out[:, 0] = (
            packed & 255
        ).to(torch.uint8)

        out[:, 1] = (
            (packed >> 8) & 255
        ).to(torch.uint8)

        out[:, 2] = (
            (packed >> 16) & 255
        ).to(torch.uint8)

        return out.flatten().contiguous()

    raise QuantizationError(
        "pack_bits supports 2, 3, 4 bits"
    )


def unpack_bits(
    data: torch.Tensor,
    bits: int,
    numel: int,
    qmin: int,
) -> torch.Tensor:

    if bits == 8:

        return (
            data
            .to(torch.int16)
            .flatten()[:numel]
        )

    d = data.flatten()

    if bits == 2:

        vals = torch.stack(
            [
                d & 3,
                (d >> 2) & 3,
                (d >> 4) & 3,
                (d >> 6) & 3,
            ],
            dim=1,
        ).flatten()

    elif bits == 4:

        vals = torch.stack(
            [
                d & 15,
                (d >> 4) & 15,
            ],
            dim=1,
        ).flatten()

    elif bits == 3:

        if len(d) % 3:
            raise QuantizationError(
                "Invalid INT3 byte length"
            )

        b = d.view(
            -1,
            3,
        ).to(torch.int32)

        packed = (
            b[:, 0]
            | (b[:, 1] << 8)
            | (b[:, 2] << 16)
        )

        vals = torch.stack(
            [
                (packed >> (3 * i)) & 7
                for i in range(8)
            ],
            dim=1,
        ).flatten()

    else:

        raise QuantizationError(
            "Unsupported bits"
        )

    return (
        vals[:numel]
        .to(torch.int16)
        + qmin
    )


def dequantize_tensor(
    qt: QuantizedTensor,
    device=None,
    dtype=torch.float32,
):

    device = device or torch.device("cpu")

    q = unpack_bits(
        qt.data.to(device),
        qt.bits,
        qt.numel,
        qt.qmin,
    ).to(dtype)

    scale = qt.scale.to(
        device=device,
        dtype=dtype,
    )

    return (
        q * scale
    ).reshape(
        qt.original_shape
    )