import time

import numpy as np
import torch

from batched_tridiagonal import (
    get_device,
    solve_batched_tridiagonal,
    synchronize,
)


def build_tridiagonal(
    batch_size: int,
    system_size: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = -torch.ones(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
    )
    diagonal = 2 * torch.ones(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
    )
    upper = -torch.ones(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
    )

    lower[:, 0] = 0
    upper[:, -1] = 0

    return lower, diagonal, upper


def solve_rows(
    values: torch.Tensor,
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """
    Решает независимые 1D системы вдоль строк.
    values имеет размер [y, x].
    """
    return solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=values,
    )


def solve_columns(
    values: torch.Tensor,
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """
    Решает независимые 1D системы вдоль столбцов.
    """
    solved_transposed = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=values.T,
    )

    return solved_transposed.T


def apply_coordinate_operator(
    p: torch.Tensor,
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """
    M(p) = A_x^(-1) p + A_y^(-1) p.

    При согласовании решений должно выполняться:

        M(p) = -b,

    где b — вклад верхнего граничного условия.
    """

    along_x = solve_rows(
        values=p,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
    )

    along_y = solve_columns(
        values=p,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
    )

    return along_x + along_y


def conjugate_gradient(
    rhs: torch.Tensor,
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    tolerance: float = 1e-6,
    max_iterations: int = 500,
) -> tuple[torch.Tensor, int, float]:
    """
    Решает M(p) = rhs методом сопряжённых градиентов.

    ВНИМАНИЕ: обычный CG верен только когда M симметрична и положительно
    определена, т.е. здесь при DX = DY = I и A = Aᵀ (B = Bᵀ). Для общих
    DX, DY или несимметричных A, B нужен CGLS
    (см. cgls в coordinator_laplace_descent.py).
    """

    p = torch.zeros_like(rhs)

    residual = rhs - apply_coordinate_operator(
        p,
        lower,
        diagonal,
        upper,
    )

    direction = residual.clone()
    residual_norm_squared = torch.sum(residual * residual)

    for iteration in range(1, max_iterations + 1):
        operator_direction = apply_coordinate_operator(
            direction,
            lower,
            diagonal,
            upper,
        )

        denominator = torch.sum(direction * operator_direction)

        alpha = residual_norm_squared / denominator

        p = p + alpha * direction
        residual = residual - alpha * operator_direction

        new_residual_norm_squared = torch.sum(residual * residual)
        residual_norm = torch.sqrt(new_residual_norm_squared).item()

        if iteration % 20 == 0 or residual_norm < tolerance:
            print(
                f"Iteration {iteration:03d} | "
                f"CG residual norm: {residual_norm:.8e}"
            )

        if residual_norm < tolerance:
            return p, iteration, residual_norm

        beta = new_residual_norm_squared / residual_norm_squared
        direction = residual + beta * direction
        residual_norm_squared = new_residual_norm_squared

    return p, max_iterations, torch.sqrt(residual_norm_squared).item()


def laplace_residual(solution: np.ndarray) -> float:
    interior = solution[1:-1, 1:-1]

    residual = (
        solution[:-2, 1:-1]
        + solution[2:, 1:-1]
        + solution[1:-1, :-2]
        + solution[1:-1, 2:]
        - 4.0 * interior
    )

    return float(np.max(np.abs(residual)))


def main() -> None:
    device = get_device()
    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64

    grid_size = 100
    interior_size = grid_size - 2

    x = torch.linspace(
        0.0,
        1.0,
        grid_size,
        device=device,
        dtype=dtype,
    )

    top_boundary = torch.sin(torch.pi * x[1:-1])

    lower, diagonal, upper = build_tridiagonal(
        batch_size=interior_size,
        system_size=interior_size,
        device=device,
        dtype=dtype,
    )

    # Вклад верхней границы в вертикальные 1D задачи.
    boundary_rhs_transposed = torch.zeros(
        (interior_size, interior_size),
        device=device,
        dtype=dtype,
    )
    boundary_rhs_transposed[:, -1] = top_boundary

    boundary_solution = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=boundary_rhs_transposed,
    ).T

    # M(p) = -boundary_solution.
    rhs = -boundary_solution

    synchronize(device)
    start = time.perf_counter()

    p, iterations, cg_residual = conjugate_gradient(
        rhs=rhs,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        tolerance=1e-6,
        max_iterations=500,
    )

    # Горизонтальное решение:
    # A_x u = -p.
    u = -solve_rows(
        values=p,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
    )

    # Вертикальное решение:
    # A_y v = p + g.
    vertical_rhs = p.T + boundary_rhs_transposed

    v = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=vertical_rhs,
    ).T

    synchronize(device)
    elapsed = time.perf_counter() - start

    difference = u - v

    # Полное 100x100 решение.
    solution = torch.zeros(
        (grid_size, grid_size),
        device=device,
        dtype=dtype,
    )
    solution[1:-1, 1:-1] = 0.5 * (u + v)
    solution[-1, :] = torch.sin(torch.pi * x)

    solution_np = solution.cpu().numpy()
    np.save("coordinator_laplace_cg_solution.npy", solution_np)

    print("\n=== Coordinator Laplace solver: Conjugate Gradient ===")
    print(f"Device: {device}")
    print(f"Grid: {grid_size} x {grid_size}")
    print(f"CG iterations: {iterations}")
    print(f"CG residual norm: {cg_residual:.8e}")
    print(f"Max |u - v|: {torch.max(torch.abs(difference)).item():.8e}")
    print(
        f"RMS |u - v|: "
        f"{torch.sqrt(torch.mean(difference * difference)).item():.8e}"
    )
    print(f"Laplace residual: {laplace_residual(solution_np):.8e}")
    print(f"Time: {elapsed:.4f} sec")
    print("Saved: coordinator_laplace_cg_solution.npy")

    try:
        direct_solution = np.load("direct_laplace_solution.npy")
        comparison = solution_np - direct_solution

        print("\n=== Comparison with direct CPU solution ===")
        print(
            f"Max |coordinator - direct|: "
            f"{np.max(np.abs(comparison)):.8e}"
        )
        print(
            f"RMS |coordinator - direct|: "
            f"{np.sqrt(np.mean(comparison * comparison)):.8e}"
        )
    except FileNotFoundError:
        print("\nDirect solution file was not found.")


if __name__ == "__main__":
    main()