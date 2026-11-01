
import torch

from src.quantization.quantizer import (
    UniformQuantizer,
    dequantize_tensor,
    pack_bits,
    unpack_bits,
)


def test_packing_round_trip(bits: int) -> None:
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    # Include boundary values and lengths that require padding.
    q = torch.tensor(
        [qmin, qmax, 0, -1, 1, qmin, qmax, 2, -2, 0, 1],
        dtype=torch.int16,
    )
    q = q.clamp(qmin, qmax)

    packed = (
        q.to(torch.int16) - qmin
        if bits != 8
        else q
    )

    if bits == 8:
        data = q.to(torch.int8)
        recovered = unpack_bits(data, bits, q.numel(), qmin)
    else:
        data = pack_bits(q, bits)
        recovered = unpack_bits(data, bits, q.numel(), qmin)

    assert torch.equal(q, recovered), (
        f"INT{bits} round-trip failed: "
        f"expected {q.tolist()}, got {recovered.tolist()}"
    )

    print(
        f"PASS INT{bits}: {q.numel()} values, "
        f"{data.numel()} stored bytes"
    )


def test_quantization(bits: int) -> None:
    torch.manual_seed(42)
    x = torch.randn(3, 17)

    quantizer = UniformQuantizer(bits)
    qt = quantizer.quantize(x)
    reconstructed = dequantize_tensor(qt)

    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()
    assert qt.bits == bits

    error = (x - reconstructed).abs().mean().item()

    print(
        f"PASS INT{bits} quantization: "
        f"shape={tuple(reconstructed.shape)}, "
        f"mean_abs_error={error:.6f}"
    )


def main() -> None:
    for bits in (2, 3, 4, 8):
        test_packing_round_trip(bits)
        test_quantization(bits)

    print("\nAll quantizer checks passed.")


if __name__ == "__main__":
    main()
