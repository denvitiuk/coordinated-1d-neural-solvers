
import time

import torch


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


device = "mps" if torch.backends.mps.is_available() else "cpu"
size = 10_000_000

print(f"Device: {device}")
print(f"Elements: {size:,}\n")

a = torch.rand(size, device=device)
b = torch.rand(size, device=device)

# Векторизованная операция:
# каждый элемент a[i] и b[i] может считаться независимо.
synchronize(device)
start = time.perf_counter()

result = a * b + 10

synchronize(device)
elapsed = time.perf_counter() - start

print(f"Vectorized PyTorch operation: {elapsed:.4f} sec")
print(f"Result device: {result.device}")
print(f"First value: {result[0].item():.4f}")