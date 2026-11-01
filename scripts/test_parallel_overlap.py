import time
import torch

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    # CPU-resident source data in pinned host memory.
    n = 16 * 1024 * 1024
    host = torch.empty(n, dtype=torch.float32, pin_memory=True)
    host.uniform_()

    # Warm up CUDA before timing.
    x = torch.randn(2048, 2048, device=device)
    for _ in range(3):
        _ = x @ x
    torch.cuda.synchronize()

    # Sequential: copy, then compute.
    start = time.perf_counter()
    a = torch.empty_like(host, device=device)
    a.copy_(host, non_blocking=False)
    y = x @ x
    torch.cuda.synchronize()
    sequential_s = time.perf_counter() - start

    # Separate streams: enqueue copy and compute independently.
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.cuda.stream(copy_stream):
        b = torch.empty_like(host, device=device)
        b.copy_(host, non_blocking=True)

    with torch.cuda.stream(compute_stream):
        z = x @ x

    # Wait for both streams before stopping the timer.
    copy_stream.synchronize()
    compute_stream.synchronize()
    parallel_s = time.perf_counter() - start

    print(f"Sequential elapsed: {sequential_s:.4f} s")
    print(f"Separate-stream elapsed: {parallel_s:.4f} s")
    print(f"Speedup: {sequential_s / parallel_s:.3f}x")
    print("NOTE: This is a synthetic test, not an LLM weight-streaming benchmark.")

if __name__ == "__main__":
    main()
