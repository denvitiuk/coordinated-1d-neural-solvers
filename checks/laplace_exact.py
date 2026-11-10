"""
Точное решение 5-точечной разностной задачи Лапласа на прямоугольнике
при u(x, ly) = top(x), остальные границы 0 — без координатора и без прогонки.

sin(kπx/lx) — точные собственные векторы разностного u_xx, поэтому
    U[i, j] = Σ_k b_k · w_k[i] · sin(kπx_j/lx),
где b — коэффициенты top в дискретном синус-базисе, а w_k решает 1D задачу по y:
    -w[i-1] + (2 + λ_k hy²) w[i] - w[i+1] = 0,  w[0] = 0,  w[ny-1] = 1,
    λ_k = (2 - 2cos(kπ hx / lx)) / hx².
"""

import math

import torch

from coordinator_laplace_descent import sine_top

DTYPE = torch.float64


def exact_discrete_solution(nx, ny, lx, ly, top=sine_top) -> torch.Tensor:
    hx, hy = lx / (nx - 1), ly / (ny - 1)
    nx_in, ny_in = nx - 2, ny - 2

    x = torch.linspace(0.0, lx, nx, dtype=DTYPE)
    k = torch.arange(1, nx_in + 1, dtype=DTYPE)
    j = torch.arange(1, nx_in + 1, dtype=DTYPE)
    sines = torch.sin(math.pi * torch.outer(j, k) / (nx - 1))  # [узел, мода]

    coeffs = torch.linalg.solve(sines, top(x[1:-1], lx))
    lam = (2.0 - 2.0 * torch.cos(math.pi * k * hx / lx)) / hx**2

    # Пакет 1D задач по y, по одной на моду (плотный solve).
    n = ny_in
    off = torch.diag(torch.ones(n - 1, dtype=DTYPE), 1)
    mats = (
        torch.diag_embed((2.0 + lam * hy**2).unsqueeze(1).expand(-1, n))
        - off
        - off.T
    )
    rhs = torch.zeros((nx_in, n, 1), dtype=DTYPE)
    rhs[:, -1, 0] = 1.0
    w = torch.linalg.solve(mats, rhs).squeeze(2)  # [мода, y]

    solution = torch.zeros((ny, nx), dtype=DTYPE)
    solution[1:-1, 1:-1] = (w * coeffs.unsqueeze(1)).T @ sines.T
    solution[-1, 1:-1] = top(x[1:-1], lx)
    return solution
