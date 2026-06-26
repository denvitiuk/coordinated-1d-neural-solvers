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
    # q хранит масштабированную функцию-координатор p = h²q.
    # Это устраняет плохое масштабирование при малом h².
    rhs_horizontal = -q

    u = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=rhs_horizontal,
    )

    # Вертикальные системы: -v_yy = q.
    # Транспонируем, чтобы каждая строка была одной системой вдоль y.
    rhs_vertical = q.T

    # Вклад верхней границы u(x, 1) = sin(pi x).
    rhs_vertical = rhs_vertical.clone()
    rhs_vertical[:, -1] += top_boundary

    v_transposed = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=rhs_vertical,
    )

    return u, v_transposed.T


def functional(
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

    # Используем сумму, не среднее:
    # для L-BFGS масштаб градиента будет удобнее.
    value = 0.5 * torch.sum(difference * difference)

    return value, u, v


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

    optimizer = torch.optim.LBFGS(
        [q],
        lr=1.0,
        max_iter=100,
        max_eval=150,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
        history_size=30,
        line_search_fn="strong_wolfe",
    )

    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls

        optimizer.zero_grad()

        value, _, _ = functional(
            q=q,
            lower=lower,
            diagonal=diagonal,
            upper=upper,
            top_boundary=top_boundary,
            h=h,
        )

        value.backward()
        closure_calls += 1

        if closure_calls % 10 == 0:
            print(f"Closure {closure_calls:02d} | J(q): {value.item():.8e}")

        return value

    synchronize(device)
    start = time.perf_counter()

    optimizer.step(closure)

    final_value, u, v = functional(
        q=q,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        top_boundary=top_boundary,
        h=h,
    )

    synchronize(device)
    elapsed = time.perf_counter() - start

    difference = u - v
    max_difference = torch.max(torch.abs(difference)).item()
    rms_difference = torch.sqrt(torch.mean(difference * difference)).item()

    torch.save(q.detach().cpu(), "coordinator_q_lbfgs.pt")
    torch.save(u.detach().cpu(), "coordinator_u_lbfgs.pt")
    torch.save(v.detach().cpu(), "coordinator_v_lbfgs.pt")

    print("\n=== Coordinator Laplace solver: L-BFGS ===")
    print(f"Device: {device}")
    print(f"Grid: {grid_size} x {grid_size}")
    print(f"Closure calls: {closure_calls}")
    print(f"Final J(q): {final_value.item():.8e}")
    print(f"Max |u - v|: {max_difference:.8e}")
    print(f"RMS |u - v|: {rms_difference:.8e}")
    print(f"Time: {elapsed:.4f} sec")
    print("Saved: coordinator_q_lbfgs.pt")
    print("Saved: coordinator_u_lbfgs.pt")
    print("Saved: coordinator_v_lbfgs.pt")


if __name__ == "__main__":
    main()