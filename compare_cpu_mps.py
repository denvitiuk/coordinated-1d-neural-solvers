import time
import torch


def benchmark(device: str, size: int = 3000) -> float:
    a = torch.rand((size, size), device=device)
    b = torch.rand((size, size), device=device)

    # Разогрев: первая операция иногда включает lazy initialization.
    _ = a @ b

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    result = a @ b

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(
        f"{device.upper():<4} | "
        f"result: {result.device} | "
        f"time: {elapsed:.3f} sec"
    )

    return elapsed


size = 3000

print(f"Matrix multiplication benchmark: {size} x {size}\n")

cpu_time = benchmark("cpu", size)

if torch.backends.mps.is_available():
    mps_time = benchmark("mps", size)

    print(f"\nMPS speedup: {cpu_time / mps_time:.2f}x")
else:
    print("\nMPS is unavailable on this machine.")