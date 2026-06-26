import time

import numpy as np


def solve_laplace_jacobi(
    nx: int = 100,
    ny: int = 100,
    tolerance: float = 1e-7,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, int, float, float]:
    """
    Решает Δu = 0 на области [0, 1] × [0, 1]
    методом Якоби с 5-точечной конечно-разностной схемой.

    Граничные условия:
        u(x, 0) = 0
        u(x, 1) = sin(pi * x)
        u(0, y) = 0
        u(1, y) = 0
    """

    x = np.linspace(0.0, 1.0, nx)

    # Текущее и следующее приближения.
    u = np.zeros((ny, nx), dtype=np.float64)
    u_next = np.zeros_like(u)

    # Верхняя граница: y = 1.
    top_boundary = np.sin(np.pi * x)
    u[-1, :] = top_boundary
    u_next[-1, :] = top_boundary

    start = time.perf_counter()
    error = float("inf")

    for iteration in range(1, max_iterations + 1):
        # 5-точечный шаблон:
        # u[i, j] = (left + right + top + bottom) / 4
        u_next[1:-1, 1:-1] = 0.25 * (
            u[:-2, 1:-1]
            + u[2:, 1:-1]
            + u[1:-1, :-2]
            + u[1:-1, 2:]
        )

        error = float(np.max(np.abs(u_next - u)))

        # Меняем ссылки, не копируя всю матрицу вручную.
        u, u_next = u_next, u

        if error < tolerance:
            break

    elapsed = time.perf_counter() - start

    return u, iteration, elapsed, error


if __name__ == "__main__":
    solution, iterations, elapsed, error = solve_laplace_jacobi()

    np.save("direct_laplace_solution.npy", solution)

    print("=== Direct 2D Laplace solver: CPU ===")
    print("Grid: 100 x 100")
    print(f"Iterations: {iterations}")
    print(f"Final error: {error:.8e}")
    print(f"Time: {elapsed:.4f} sec")
    print(f"Min value: {solution.min():.6f}")
    print(f"Max value: {solution.max():.6f}")
    print("Saved: direct_laplace_solution.npy")