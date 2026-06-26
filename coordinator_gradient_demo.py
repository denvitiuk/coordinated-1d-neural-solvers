import torch

from batched_tridiagonal import get_device, solve_batched_tridiagonal


def build_laplace_1d_system(
    batch_size: int,
    system_size: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Трёхточечная схема для -u'' = f:
        -u[i-1] + 2u[i] - u[i+1] = f[i]
    """

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


def solve_family(
    q: torch.Tensor,
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    sign: float,
) -> torch.Tensor:
    """
    Решает семейство независимых 1D систем.

    sign = -1: условно горизонтальные задачи.
    sign = +1: условно вертикальные задачи.
    """

    rhs = sign * q

    return solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=rhs,
    )


def main() -> None:
    device = get_device()
    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64

    batch_size = 98
    system_size = 98

    lower, diagonal, upper = build_laplace_1d_system(
        batch_size=batch_size,
        system_size=system_size,
        device=device,
        dtype=dtype,
    )

    # q — функция-координатор на внутренних узлах.
    q = torch.zeros(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    u = solve_family(
        q=q,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        sign=-1.0,
    )

    v = solve_family(
        q=q,
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        sign=1.0,
    )

    residual = u - v
    functional = 0.5 * torch.sum(residual * residual)

    functional.backward()

    print("=== Coordinator functional demo ===")
    print(f"Device: {device}")
    print(f"q shape: {tuple(q.shape)}")
    print(f"u shape: {tuple(u.shape)}")
    print(f"v shape: {tuple(v.shape)}")
    print(f"J(q): {functional.item():.8e}")
    print(f"Gradient norm: {torch.linalg.vector_norm(q.grad).item():.8e}")
    print(f"Max |gradient|: {torch.max(torch.abs(q.grad)).item():.8e}")


if __name__ == "__main__":
    main()