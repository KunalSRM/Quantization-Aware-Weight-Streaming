
import torch
import torch.nn as nn

from src.streaming.streamed_linear import StreamedLinear


def test_streamed_linear(device: torch.device, bits: int) -> None:
    torch.manual_seed(42)

    original = nn.Linear(32, 16, bias=True).to(device)
    streamed = StreamedLinear(original, bits).to(device)

    x = torch.randn(4, 32, device=device)
    output = streamed(x)

    assert output.shape == (4, 16), (
        f"Unexpected output shape: {output.shape}"
    )
    assert torch.isfinite(output).all(), (
        "Output contains NaN or Inf"
    )

    print(
        f"PASS INT{bits} | device={device} | "
        f"shape={tuple(output.shape)}"
    )


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Testing StreamedLinear on: {device}")

    for bits in (2, 3, 4, 8):
        test_streamed_linear(device, bits)

    print("\nAll StreamedLinear smoke tests passed.")


if __name__ == "__main__":
    main()
