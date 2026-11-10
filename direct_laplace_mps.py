import time

import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def solve_laplace_jacobi(
    nx: int = 100,
    ny: int = 100,
    tolerance: float = 1e-6,
    max_iterations: int = 100_000,
) -> tuple[torch.Tensor, int, float, float, str]:
    """
    То же уравнение Лапласа и те же граничные условия,
    что в direct_laplace_cpu.py, но на MPS/CUDA/CPU через PyTorch.
    """

    device = get_device()
    # Apple MPS does not support float64; use float32 on MPS/CUDA.
    # Keep float64 only for the CPU fallback.
    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64

    x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)

    u = torch.zeros((ny, nx), device=device, dtype=dtype)
    u_next = torch.zeros_like(u)

    top_boundary = torch.sin(torch.pi * x)
    u[-1, :] = top_boundary
    u_next[-1, :] = top_boundary

    synchronize(device)
    start = time.perf_counter()
    error = float("inf")

    for iteration in range(1, max_iterations + 1):
        u_next[1:-1, 1:-1] = 0.25 * (
            u[:-2, 1:-1]
            + u[2:, 1:-1]
            + u[1:-1, :-2]
            + u[1:-1, 2:]
        )

        error = torch.max(torch.abs(u_next - u)).item()

        u, u_next = u_next, u

        if error < tolerance:
            break

    synchronize(device)
    elapsed = time.perf_counter() - start

    return u, iteration, elapsed, error, device


if __name__ == "__main__":
    solution, iterations, elapsed, error, device = solve_laplace_jacobi()

    torch.save(solution.cpu(), "direct_laplace_mps_solution.pt")

    print("=== Direct 2D Laplace solver: PyTorch ===")
    print(f"Device: {device}")
    print("Grid: 100 x 100")
    print(f"Iterations: {iterations}")
    print(f"Final error: {error:.8e}")
    print(f"Time: {elapsed:.4f} sec")
    print(f"Min value: {solution.min().item():.6f}")
    print(f"Max value: {solution.max().item():.6f}")
    print("Saved: direct_laplace_mps_solution.pt")