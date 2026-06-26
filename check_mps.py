import time

import torch


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


device = choose_device()

print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")

size = 3000

a = torch.rand((size, size), device=device)
b = torch.rand((size, size), device=device)

start = time.perf_counter()

result = a @ b

# На GPU вычисления запускаются асинхронно.
# Без synchronize() замер времени был бы неправильным.
if device == "mps":
    torch.mps.synchronize()
elif device == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print(f"Matrix size: {size} x {size}")
print(f"Result device: {result.device}")
print(f"Time: {elapsed:.3f} seconds")
print(f"First value: {result[0, 0].item():.4f}")