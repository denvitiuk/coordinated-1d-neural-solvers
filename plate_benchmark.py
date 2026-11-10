"""
Замер оптимизации задачи о пластине (float64, CPU, плюс лучший вариант на MPS).

Варианты:
  CR + цикл прогонки      — как было до оптимизации;
  CR + BLAS-прогонка      — та же математика, быстрее одна прогонка;
  GMRES + ADI + BLAS      — меньше итераций (предобусловливание) и быстрые прогонки;
  прямое (собств. векторы)— эталон exact_plate, не координация, для масштаба.

Пишет results/plate_benchmark.csv и печатает таблицу в markdown.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from batched_tridiagonal import synchronize
from coordinator_laplace_descent import (
    AdiPreconditioner,
    CoordinatorProblem,
    conjugate_residual,
    gmres,
)
from plate_coordination import Plate, assemble_plate, build_plate_problem, exact_plate

GRIDS = [(51, 121), (101, 241), (201, 481)]
RTOL = 1e-10
REPEATS = 3


def timed(fn, device="cpu"):
    best, out = float("inf"), None
    for _ in range(REPEATS):
        synchronize(device)
        start = time.perf_counter()
        out = fn()
        synchronize(device)
        best = min(best, time.perf_counter() - start)
    return best, out


def variant(plate, solver, use_adi, device="cpu", dtype=torch.float64, rtol=RTOL):
    def run():
        base = build_plate_problem(plate, device, dtype)
        pr = CoordinatorProblem(base.a, base.b, base.dx, base.dy, base.fx, base.fy, solver=solver)
        q0 = torch.zeros_like(pr.fx)
        if use_adi:
            res = gmres(pr, q0, rtol=rtol, preconditioner=AdiPreconditioner(pr))
        else:
            res = conjugate_residual(pr, q0, rtol=rtol, max_iter=20_000)
        return pr, res

    seconds, (pr, res) = timed(run, device)
    field = assemble_plate(plate, pr, res.q).cpu().double()
    return seconds, res, field


def main() -> None:
    rows = []
    for mx, ny in GRIDS:
        plate = Plate(mx=mx, ny=ny)
        t_exact, exact = timed(lambda: exact_plate(plate))
        cases = [
            ("CR + цикл прогонки (было)", "thomas", False, "cpu", torch.float64),
            ("CR + BLAS-прогонка", "dense", False, "cpu", torch.float64),
            ("GMRES + ADI + BLAS (стало)", "dense", True, "cpu", torch.float64),
        ]
        if torch.backends.mps.is_available():
            cases.append(("GMRES + ADI + BLAS, MPS float32", "dense", True, "mps", torch.float32))
        for name, solver, adi, device, dtype in cases:
            rtol = RTOL if dtype == torch.float64 else 1e-6
            sec, res, field = variant(plate, solver, adi, device, dtype, rtol)
            rows.append({
                "grid": f"{mx}x{ny}", "unknowns": (mx - 2) * (ny - 2), "variant": name,
                "iterations": res.iterations, "solves": res.tridiagonal_solves,
                "seconds": sec, "max_err": torch.max(torch.abs(field - exact)).item(),
            })
        rows.append({
            "grid": f"{mx}x{ny}", "unknowns": (mx - 2) * (ny - 2),
            "variant": "прямое решение (не координация)", "iterations": "", "solves": "",
            "seconds": t_exact, "max_err": 0.0,
        })

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "plate_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("| Сетка | Неизвестных | Вариант | Итераций | Прогонок | Время, с | max abs(T − T*) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        err = f"{r['max_err']:.1e}" if r["iterations"] != "" else "—"
        print(f"| {r['grid']} | {r['unknowns']} | {r['variant']} | {r['iterations']} "
              f"| {r['solves']} | {r['seconds']:.3f} | {err} |")


if __name__ == "__main__":
    main()
