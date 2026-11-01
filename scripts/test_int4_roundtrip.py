
import torch

from src.quantization.quantizer import (
    UniformQuantizer,
    unpack_bits,
)

def main():
    bits = 4
    quantizer = UniformQuantizer(bits)

    # Test values include negative, zero and positive values.
    original = torch.tensor(
        [-8.0, -7.0, -3.0, -1.0, 0.0, 1.0, 3.0, 7.0]
    )

    quantized = quantizer.quantize(original)

    recovered = unpack_bits(
        quantized.data,
        quantized.bits,
        quantized.numel,
        quantized.qmin,
    )

    expected = torch.round(
        original / quantized.scale
    ).clamp(-8, 7).to(torch.int16)

    print("Original values: ", original.tolist())
    print("Packed bytes:    ", quantized.data.tolist())
    print("Expected INT4:   ", expected.tolist())
    print("Recovered INT4:  ", recovered.tolist())

    if torch.equal(expected, recovered):
        print("\nPASS: INT4 packing/unpacking is correct.")
    else:
        print("\nFAIL: INT4 packing/unpacking mismatch.")

if __name__ == "__main__":
    main()
