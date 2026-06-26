import time

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


def solve_coordinated_families(
    q: torch.Tensor,
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    top_boundary: torch.Tensor,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q имеет размер [98, 98].

    Горизонтальные задачи:
        -u_xx = -q

    Вертикальные задачи:
        -v_yy = q

    u и v должны совпасть после оптимального выбора q.
    """

    h_squared = h * h

    # 98 горизонтальных систем.
    rhs_horizontal = -h_squared * q

    horizontal_solution = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=rhs_horizontal,
    )

    # 98 вертикальных систем.
    # Каждая строка после transpose — отдельная система вдоль y.
    vertical_boundary = torch.zeros_like(q.T)

    # Последний внутренний узел по y связан с верхней границей y = 1.
    vertical_boundary[:, -1] = top_boundary

    rhs_vertical = h_squared * q.T + vertical_boundary

    vertical_solution_transposed = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=rhs_vertical,
    )

    vertical_solution = vertical_solution_transposed.T

    return horizontal_solution, vertical_solution


def functional_value(
    q: torch.Tensor,
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    top_boundary: torch.Tensor,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, v = solve_coordinated_families(
        q=q,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        top_boundary=top_boundary,
        h=h,
    )

    difference = u - v

    # Среднее, а не сумма: значения не зависят от размера сетки.
    functional = 0.5 * torch.mean(difference * difference)

    return functional, u, v


def main() -> None:
    device = get_device()
    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64

    grid_size = 100
    interior_size = grid_size - 2
    h = 1.0 / (grid_size - 1)

    x = torch.linspace(
        0.0,
        1.0,
        grid_size,
        device=device,
        dtype=dtype,
    )

    # Верхняя граница без угловых точек.
    top_boundary = torch.sin(torch.pi * x[1:-1])

    lower, diagonal, upper = build_tridiagonal(
        batch_size=interior_size,
        system_size=interior_size,
        device=device,
        dtype=dtype,
    )

    q = torch.zeros(
        (interior_size, interior_size),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    max_iterations = 30
    initial_step = 1.0

    synchronize(device)
    start = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        functional, _, _ = functional_value(
            q=q,
            lower=lower,
            diagonal=diagonal,
            upper=upper,
            top_boundary=top_boundary,
            h=h,
        )

        functional.backward()

        gradient = q.grad
        gradient_norm = torch.sqrt(torch.sum(gradient * gradient))

        # Нормированный градиент: шаг не зависит от масштаба gradient.
        direction = gradient / (gradient_norm + 1e-12)

        current_value = functional.item()
        step = initial_step

        # Backtracking line search.
        with torch.no_grad():
            while step > 1e-8:
                candidate_q = q - step * direction

                candidate_value, _, _ = functional_value(
                    q=candidate_q,
                    lower=lower,
                    diagonal=diagonal,
                    upper=upper,
                    top_boundary=top_boundary,
                    h=h,
                )

                if candidate_value.item() < current_value:
                    break

                step *= 0.5

        q = candidate_q.detach().requires_grad_(True)

        print(
            f"Iteration {iteration:02d} | "
            f"J(q): {current_value:.8e} | "
            f"||grad||: {gradient_norm.item():.8e} | "
            f"step: {step:.3e}"
        )

        if gradient_norm.item() < 1e-6:
            break

    final_functional, u, v = functional_value(
        q=q,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        top_boundary=top_boundary,
        h=h,
    )

    synchronize(device)
    elapsed = time.perf_counter() - start

    torch.save(q.detach().cpu(), "coordinator_q.pt")
    torch.save(u.detach().cpu(), "coordinator_horizontal.pt")
    torch.save(v.detach().cpu(), "coordinator_vertical.pt")

    max_difference = torch.max(torch.abs(u - v)).item()

    print("\n=== Coordinator method for Laplace equation ===")
    print(f"Device: {device}")
    print(f"Grid: {grid_size} x {grid_size}")
    print(f"Iterations: {iteration}")
    print(f"Final J(q): {final_functional.item():.8e}")
    print(f"Max |u - v|: {max_difference:.8e}")
    print(f"Time: {elapsed:.4f} sec")
    print("Saved: coordinator_q.pt")
    print("Saved: coordinator_horizontal.pt")
    print("Saved: coordinator_vertical.pt")


if __name__ == "__main__":
    main()