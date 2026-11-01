
import torch
import torch.nn as nn

from src.streaming.streamed_linear import StreamedLinear


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create the original layer on the selected device.
    original = nn.Linear(128, 64).to(device).eval()

    # StreamedLinear quantizes the weights and stores them on CPU.
    quantized = StreamedLinear(original, bits=4).eval()

    # Generate input on the same device as the original layer.
    x = torch.randn(4, 128, device=device)

    with torch.no_grad():
        expected = original(x)
        actual = quantized(x)

    mae = (expected - actual).abs().mean().item()
    max_error = (expected - actual).abs().max().item()

    print("\n===== INT4 Layer Comparison =====")
    print(f"Device: {device}")
    print(f"Original output shape: {tuple(expected.shape)}")
    print(f"Quantized output shape: {tuple(actual.shape)}")
    print(f"Mean absolute error: {mae:.6f}")
    print(f"Maximum absolute error: {max_error:.6f}")

    if not torch.isfinite(actual).all():
        print("FAIL: Quantized output contains NaN or Inf.")
        return

    if expected.shape != actual.shape:
        print("FAIL: Output shapes do not match.")
        return

    print("PASS: Quantized output is finite and shapes match.")


if __name__ == "__main__":
    main()
