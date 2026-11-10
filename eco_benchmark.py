"""
Замер: экологическая модель (переменные коэффициенты), CPU float64, rtol 1e-10.

  GMRES + ADI          — координация с предобусловливанием (основной метод);
  GMRES без ADI        — координация без предобусловливания (как в дипломе, только Крылов);
  прямое (плотная LU)  — двумерная система целиком, только малые сетки.

Пишет results/eco_benchmark.csv и печатает таблицу.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from eco_pollution import EcoSolver, Region, build_operators, dense_operator

GRIDS = [(101, 51), (201, 101), (401, 201), (801, 401)]
DENSE_MAX = 6000  # неизвестных: плотная LU дальше не влезает разумно
NO_PRECOND_MAX = 100_000
RTOL = 1e-10


def main() -> None:
    rows = []
    for nx, ny in GRIDS:
        region = Region(nx=nx, ny=ny)
        ops = build_operators(region)
        n = (nx - 2) * (ny - 2)

        t0 = time.perf_counter()
        solver = EcoSolver(ops)
        c, res = solver.solve(ops.source, rtol=RTOL)
        total = time.perf_counter() - t0
        resid = (torch.linalg.vector_norm(ops.apply_l(c) - ops.source)
                 / torch.linalg.vector_norm(ops.source)).item()
        rows.append(dict(grid=f"{nx}x{ny}", unknowns=n, variant="GMRES + ADI",
                         iterations=res.iterations, seconds=total, residual=resid))

        if n <= NO_PRECOND_MAX:
            t0 = time.perf_counter()
            c2, r2 = EcoSolver(ops, adi_steps=0).solve(ops.source, rtol=RTOL, max_iter=5000)
            total = time.perf_counter() - t0
            resid2 = (torch.linalg.vector_norm(ops.apply_l(c2) - ops.source)
                      / torch.linalg.vector_norm(ops.source)).item()
            rows.append(dict(grid=f"{nx}x{ny}", unknowns=n, variant="GMRES без ADI",
                             iterations=r2.iterations, seconds=total, residual=resid2))

        if n <= DENSE_MAX:
            t0 = time.perf_counter()
            direct = torch.linalg.solve(dense_operator(ops), ops.source.reshape(-1))
            total = time.perf_counter() - t0
            rows.append(dict(grid=f"{nx}x{ny}", unknowns=n, variant="прямое (плотная LU)",
                             iterations="", seconds=total,
                             residual=(torch.max(torch.abs(direct.reshape(c.shape) - c))
                                       / torch.max(direct)).item()))

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "eco_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n| Сетка | Неизвестных | Метод | Итераций | Время, с | ‖Lc − S‖/‖S‖ |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['grid']} | {r['unknowns']} | {r['variant']} | {r['iterations']} "
              f"| {r['seconds']:.3f} | {r['residual']:.1e} |")


if __name__ == "__main__":
    main()
