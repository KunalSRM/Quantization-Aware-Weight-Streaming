
import torch
import torch.nn as nn

from src.streaming.apply import apply_streaming


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = torch.device("cuda")

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )

    count = apply_streaming(model, bits=4, device=device)

    x = torch.randn(8, 32, device=device)

    with torch.no_grad():
        output = model(x)

    assert count == 2, f"Expected 2 replaced layers, got {count}"
    assert output.shape == (8, 16), output.shape
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    print(f"Replaced linear layers: {count}")
    print(f"Output device: {output.device}")
    print(f"Output shape: {tuple(output.shape)}")
    print("PASS: apply_streaming integration smoke test")


if __name__ == "__main__":
    main()
