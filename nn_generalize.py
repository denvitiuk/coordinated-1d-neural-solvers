"""
Опыт 3 ниши «подзадачи — нейросети»: обобщение и предобусловливание.

Берём сеть из опыта 2 (results/nn_line_solver_bnd.pt, обучена ТОЛЬКО на одномерных
задачах) и, не дообучая, решаем ею двумерные задачи, которых она не видела:
новые поля k(x, y), более контрастные k, другие граничные условия.

Предобусловливатель «грубая модель»: сеть остаётся чёрным ящиком, а для ускорения
GMRES берётся тот же P = Y₀ · ADI · X₀, что в координации с прогонкой, но для
ПОСТОЯННОГО коэффициента k̄ (среднее k). Его можно построить, не зная, что внутри
сети; он меняет только число итераций, но не ответ.

    python nn_generalize.py
"""

from __future__ import annotations

import math
import time

import torch

import nn_coordination as nc
from adi_direct_vs_coordination import DirectAdi

M, H = nc.M, nc.H


def make_problem(seed: int, contrast: float, bc: str):
    g = torch.Generator().manual_seed(seed)
    ks = torch.randn(6, generator=g, dtype=torch.float64)

    def kfun(x, y):
        return torch.exp(contrast * (ks[0] * torch.sin(math.pi * x) * torch.sin(math.pi * y)
                                     + ks[1] * torch.cos(2 * math.pi * x) + ks[2] * torch.cos(2 * math.pi * y)
                                     + 0.5 * ks[3] * torch.sin(3 * math.pi * x + ks[4]) * torch.cos(math.pi * y + ks[5])))

    nodes = torch.arange(1, M + 1, dtype=torch.float64) * H
    halves = (torch.arange(0, M + 1, dtype=torch.float64) + 0.5) * H
    kx = kfun(halves[None, :], nodes[:, None])
    ky = kfun(nodes[:, None], halves[None, :])
    gx = torch.zeros(M, M, dtype=torch.float64)
    gy = torch.zeros(M, M, dtype=torch.float64)
    if bc == "top=1":
        gy[-1, :] = ky[:, -1]
    elif bc == "top=sin":
        gy[-1, :] = ky[:, -1] * torch.sin(math.pi * nodes)
    elif bc == "right=1, bottom=y-профиль":
        gx[:, -1] = kx[:, -1]                                   # правая сторона φ = 1
        gy[0, :] = ky[:, 0] * 0.5 * (1 - torch.cos(2 * math.pi * nodes))  # снизу горб
    return kx, ky, gx, gy


class CoarsePreconditioner:
    """P ≈ (X⁻¹ + Y⁻¹)⁻¹ = Y (X + Y)⁻¹ X для постоянного k̄: k̄ · T_y · ADI(T_x z)."""

    def __init__(self, kbar: float) -> None:
        self.kbar = kbar
        self.tridiagonal_solves = 0
        self.b = torch.zeros(M, M, dtype=torch.float64)   # DirectAdi берёт отсюда device/dtype
        self.adi = DirectAdi(self, M, "thomas")

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        tx = self.adi._t(z)                  # T вдоль x (последняя ось)
        w = self.adi(tx)                     # ≈ (T_x + T_y)⁻¹
        return self.kbar * self.adi._t(w.T).T


def run_case(solver, seed, contrast, bc, precond):
    kx, ky, gx, gy = make_problem(seed, contrast, bc)
    ref = nc.reference(kx, ky, gx, gy)
    pre = CoarsePreconditioner(0.5 * (kx.mean() + ky.mean()).item()) if precond else None
    t0 = time.perf_counter()
    phi, rel_res, inner, *_ = nc.coordinate(solver, kx, ky, gx, gy, preconditioner=pre)
    dt = time.perf_counter() - t0
    err = (torch.linalg.vector_norm(phi - ref) / torch.linalg.vector_norm(ref)).item()
    kmin, kmax = min(kx.min(), ky.min()).item(), max(kx.max(), ky.max()).item()
    return err, inner, dt, kmax / kmin, rel_res


def main() -> None:
    solver = nc.LineSolver()
    solver.load_state_dict(torch.load("results/nn_line_solver_bnd.pt"))
    solver.eval()

    print("A. Предобусловливатель (поле из опыта 2, seed 7):")
    for precond in (False, True):
        err, inner, dt, ratio, res = run_case(solver, 7, 0.6, "top=1", precond)
        print(f"   {'грубый ADI' if precond else 'без P     '}: ошибка φ {err:.2%}, итераций GMRES {inner:5d}, "
              f"{dt:5.1f} с, невязка {res:.0e}", flush=True)

    print("\nB. Новые задачи (сеть не дообучалась), с грубым ADI:")
    print(f"   {'seed':>4} | {'контраст':>8} | {'k_max/k_min':>11} | {'граничные условия':26} | "
          f"{'ошибка φ':>8} | {'итер.':>5} | {'время':>6}")
    errs = []
    for seed in (11, 12, 13, 14, 15):
        for contrast in (0.6, 1.0):
            for bc in ("top=1", "top=sin", "right=1, bottom=y-профиль"):
                err, inner, dt, ratio, res = run_case(solver, seed, contrast, bc, True)
                errs.append((contrast, bc, err))
                print(f"   {seed:>4} | {contrast:8.1f} | {ratio:11.1f} | {bc:26} | {err:8.2%} | {inner:5d} | "
                      f"{dt:5.1f} с", flush=True)
    print("\nСредняя ошибка φ:")
    for contrast in (0.6, 1.0):
        for bc in ("top=1", "top=sin", "right=1, bottom=y-профиль"):
            e = [x[2] for x in errs if x[0] == contrast and x[1] == bc]
            print(f"   контраст {contrast}, {bc:26}: {sum(e) / len(e):.2%} (макс {max(e):.2%})")


if __name__ == "__main__":
    main()
