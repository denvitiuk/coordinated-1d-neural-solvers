"""
Контрольный опыт: даёт ли что-то сама координация, или всё ускорение — от ADI?

Предобусловливатель метода координации P = Y · ADI_m · X, где ADI_m ≈ (X + Y)⁻¹ —
приближённое решение исходной двумерной задачи. С точным (X + Y)⁻¹ было бы M P = I.
Поэтому сравниваем с обычным подходом: GMRES с тем же ADI_m (те же τ, те же 6 шагов,
та же прогонка) прямо для пятиточечной схемы (X + Y) φ = b, без координатора.

    python adi_direct_vs_coordination.py            # CPU float64 + GPU float32
"""

from __future__ import annotations

import math
import sys
import time

import torch

from batched_tridiagonal import TridiagonalFactor, get_device, synchronize
from coordinator_laplace_descent import AdiPreconditioner, build_laplace_problem, gmres
from laplace_gpu_benchmark import _laplace5, exact_discrete, solve_coordination, top_values

STEPS = 6


class DirectProblem:
    """(X + Y) φ = b в интерфейсе, который ждёт gmres(): residual = b − Lφ, apply_m = L."""

    def __init__(self, n: int, device: str, dtype: torch.dtype) -> None:
        top = top_values(torch.linspace(0, 1, n, device=device, dtype=dtype))[1:-1]
        self.zero_top = torch.zeros_like(top)
        self.b = -_laplace5(torch.zeros((n - 2, n - 2), device=device, dtype=dtype), top)
        self.tridiagonal_solves = 0

    def apply_m(self, v: torch.Tensor) -> torch.Tensor:
        return _laplace5(v, self.zero_top)

    def residual(self, phi: torch.Tensor) -> torch.Tensor:
        return self.b - self.apply_m(phi)


class DirectAdi:
    """ADI_m ≈ (X + Y)⁻¹, X = tridiag(−1, 2, −1) по строкам, Y — по столбцам."""

    def __init__(self, problem: DirectProblem, m: int, solver: str, steps: int = STEPS) -> None:
        self.problem = problem
        dev, dt = problem.b.device, problem.b.dtype
        ones = torch.ones((m, m), device=dev, dtype=dt)
        lower, upper = -ones.clone(), -ones.clone()
        lower[:, 0] = 0.0
        upper[:, -1] = 0.0
        self.coef = (lower, 2 * ones, upper)
        k = torch.arange(1, m + 1, dtype=torch.float64)
        eig = 4 * torch.sin(k * math.pi / (2 * (m + 1))) ** 2
        alpha, beta = eig.min().item(), eig.max().item()
        # Те же геометрические параметры, что в AdiPreconditioner.
        self.taus = [beta * (alpha / beta) ** ((2 * j - 1) / (2 * steps)) for j in range(1, steps + 1)]
        self.factors = [TridiagonalFactor(lower, 2 * ones + tau, upper, method=solver) for tau in self.taus]

    def _t(self, w):  # трёхдиагональное умножение вдоль последней оси
        lower, diagonal, upper = self.coef
        y = diagonal * w
        y[:, 1:] += lower[:, 1:] * w[:, :-1]
        y[:, :-1] += upper[:, :-1] * w[:, 1:]
        return y

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        w = torch.zeros_like(f)
        for tau, fac in zip(self.taus, self.factors):
            rhs = f - self._t(w.T).T + tau * w          # (X + τ) w½ = f − (Y − τ) w
            self.problem.tridiagonal_solves += 1
            w_half = fac.solve(rhs)
            rhs = f - self._t(w_half) + tau * w_half     # (Y + τ) w = f − (X − τ) w½
            self.problem.tridiagonal_solves += 1
            w = fac.solve(rhs.T).T
        return w


def solve_direct(n, device, dtype, rtol, solver):
    pr = DirectProblem(n, device, dtype)
    res = gmres(pr, torch.zeros_like(pr.b), rtol=rtol, max_iter=500,
                preconditioner=DirectAdi(pr, n - 2, solver))
    return res.q, res.iterations, pr.tridiagonal_solves


def timed(fn, *args, repeats=3):
    out = fn(*args)  # прогрев
    best = math.inf
    for _ in range(repeats):
        synchronize(args[1])
        t0 = time.perf_counter()
        out = fn(*args)
        synchronize(args[1])
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    gpu = get_device()
    cases = [("cpu", torch.float64, 1e-10, "auto"), ("cpu", torch.float64, 1e-10, "pcr")]
    if gpu != "cpu":
        prec = torch.float64 if gpu == "cuda" else torch.float32
        cases.append((gpu, prec, 1e-10 if prec == torch.float64 else 1e-5, "pcr"))
    grids = [int(a) for a in sys.argv[1:]] or [100, 200, 400, 800]
    print(f"{'сетка':>9} | {'устройство':14} | {'координация':>26} | {'GMRES+ADI на исходной':>26}")
    for n in grids:
        exact = exact_discrete(n)[1:-1, 1:-1]
        for device, dtype, rtol, solver in cases:
            reps = 3 if n <= 400 else 1
            tc, (phi_c, it_c, s_c) = timed(
                lambda m, d: solve_coordination(f"adi:{solver}", m, d, dtype, rtol), n, device, repeats=reps)
            td, (phi_d, it_d, s_d) = timed(
                lambda m, d: solve_direct(m, d, dtype, rtol, solver), n, device, repeats=reps)
            ec = (phi_c.cpu().double() - exact).abs().max().item()
            ed = (phi_d.cpu().double() - exact).abs().max().item()
            tag = f"{device} {str(dtype)[-7:]} {solver}"
            print(f"{n:>4}x{n:<4} | {tag:14} | {tc:7.3f} с {it_c:3d} it {s_c:4d} пр. {ec:.0e} | "
                  f"{td:7.3f} с {it_d:3d} it {s_d:4d} пр. {ed:.0e}", flush=True)


if __name__ == "__main__":
    main()
