"""Общая тестовая задача (float64, CPU) и её плотный эталон."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from coordinator_laplace_descent import CoordinatorProblem, LinearMap  # noqa: E402

DTYPE = torch.float64
ROWS, COLS = 6, 7  # неквадратная сетка ловит перепутанные строки/столбцы


def tridiagonal(
    batch: int, size: int, lower: float, diagonal: float, upper: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lo = torch.full((batch, size), lower, dtype=DTYPE)
    di = torch.full((batch, size), diagonal, dtype=DTYPE)
    up = torch.full((batch, size), upper, dtype=DTYPE)
    lo[:, 0] = 0
    up[:, -1] = 0
    return lo, di, up


def dense_tridiagonal(lo: torch.Tensor, di: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Плотная матрица одной системы (1D коэффициенты)."""
    return torch.diag(di) + torch.diag(lo[1:], -1) + torch.diag(up[:-1], 1)


def build_general_problem(seed: int = 0) -> CoordinatorProblem:
    """Несимметричные A, B; DX диагональная ≠ I; DY = I + 0.1·randn."""
    gen = torch.Generator().manual_seed(seed)
    n = ROWS * COLS

    dx = 1.0 + 0.5 * torch.rand((ROWS, COLS), generator=gen, dtype=DTYPE)
    dy = torch.eye(n, dtype=DTYPE) + 0.1 * torch.randn((n, n), generator=gen, dtype=DTYPE)

    return CoordinatorProblem(
        a=tridiagonal(ROWS, COLS, -1.0, 4.0, -0.5),
        b=tridiagonal(COLS, ROWS, -0.3, 3.0, -1.2),
        dx=LinearMap(diagonal=dx),
        dy=LinearMap(dense=dy),
        fx=torch.randn((ROWS, COLS), generator=gen, dtype=DTYPE),
        fy=torch.randn((ROWS, COLS), generator=gen, dtype=DTYPE),
    )


def dense_reference(problem: CoordinatorProblem) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Плотные M и c для r = c - M q над q.reshape(-1) (row-major),
    собранные независимо от прогонки.
    """
    rows, cols = problem.fx.shape
    n = rows * cols

    a_full = torch.block_diag(
        *[dense_tridiagonal(*(t[i] for t in problem.a)) for i in range(rows)]
    )

    # B действует на столбцы: блочно-диагональна в column-major порядке.
    b_block = torch.block_diag(
        *[dense_tridiagonal(*(t[j] for t in problem.b)) for j in range(cols)]
    )
    perm = torch.arange(n).reshape(rows, cols).T.reshape(-1)  # col-major ← row-major
    p = torch.eye(n, dtype=DTYPE)[perm]
    b_full = p.T @ b_block @ p

    def dense_of(op: LinearMap) -> torch.Tensor:
        if op.dense is not None:
            return op.dense
        if op.diagonal is not None:
            return torch.diag(op.diagonal.reshape(-1))
        return op.scale * torch.eye(n, dtype=DTYPE)

    m = torch.linalg.solve(a_full, dense_of(problem.dx)) + torch.linalg.solve(
        b_full, dense_of(problem.dy)
    )
    c = torch.linalg.solve(a_full, problem.fx.reshape(-1)) - torch.linalg.solve(
        b_full, problem.fy.reshape(-1)
    )
    return m, c


def exact_minimizer(problem: CoordinatorProblem) -> torch.Tensor:
    m, c = dense_reference(problem)
    q_star = torch.linalg.lstsq(m, c.unsqueeze(1)).solution.squeeze(1)
    return q_star.reshape(problem.fx.shape)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()
