
import torch
import torch.nn as nn

from src.streaming.streamed_linear import StreamedLinear


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = torch.device("cuda")
    torch.manual_seed(42)

    original = nn.Linear(128, 64, bias=True).to(device).eval()
    x = torch.randn(16, 128, device=device)

    with torch.no_grad():
        reference = original(x)

        for bits in (2, 3, 4, 8):
            streamed = StreamedLinear(original, bits).to(device).eval()
            actual = streamed(x)

            mae = (reference - actual).abs().mean().item()
            max_error = (reference - actual).abs().max().item()
            relative_error = (
                (reference - actual).norm()
                / reference.norm().clamp_min(1e-12)
            ).item()

            print(
                f"INT{bits}: MAE={mae:.6f}, "
                f"MaxError={max_error:.6f}, "
                f"RelativeError={relative_error:.6f}"
            )

    print("\nAccuracy comparison completed.")


if __name__ == "__main__":
    main()
