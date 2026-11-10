"""
Финальный прогон Лапласа (diag 2, -1, -1) на квадрате и прямоугольниках, float64, CPU.

Два граничных условия сверху:
  sin  — sin(πx/lx): одна гармоника, задача фактически одномерная (лёгкий случай);
  const — 1: все нечётные гармоники (честный тест скорости).
Эталон — точное разностное решение (checks/laplace_exact.py).
"""

import common  # noqa: F401  (добавляет корень проекта в sys.path)

import torch

from coordinator_laplace_descent import (
    assemble_solution,
    build_laplace_problem,
    cgls,
    conjugate_residual,
    gmres,
    sine_top,
    steepest_descent,
)
from laplace_exact import DTYPE, exact_discrete_solution

RTOL = 1e-10
MAX_ITER = 3000  # спуску на const хватит не всегда — это и есть результат

CASES = [
    (100, 100, 1.0, 1.0),  # квадрат
    (121, 61, 2.0, 1.0),   # прямоугольник, hx = hy
    (101, 41, 2.0, 1.0),   # прямоугольник, hx ≠ hy
    (41, 121, 1.0, 3.0),   # вытянутый по y
]

TOPS = {
    "sin": sine_top,
    "const": lambda x, lx: torch.ones_like(x),
}

METHODS = {
    "SD": steepest_descent,
    "CGLS": cgls,
    "CR": conjugate_residual,
    "GMRES": gmres,
}


def run_case(nx, ny, lx, ly, top_name):
    top = TOPS[top_name]
    problem = build_laplace_problem(nx, ny, "cpu", DTYPE, lx=lx, ly=ly, top=top)
    q0 = torch.zeros_like(problem.fx)
    exact = exact_discrete_solution(nx, ny, lx, ly, top=top)

    rows = []
    for name, solver in METHODS.items():
        res = solver(problem, q0, rtol=RTOL, max_iter=MAX_ITER)
        err = torch.max(torch.abs(assemble_solution(problem, res.q, lx, top) - exact)).item()
        rows.append((name, res, err))
    return rows


def run_laplace() -> list[dict]:
    print(f"=== Laplace on rectangles, rtol = {RTOL:g}, max_iter = {MAX_ITER} ===")
    print(f"{'top':>5} {'grid x*y':>9} {'domain':>7} | {'method':>6} {'iters':>6} "
          f"{'solves':>7} {'time, s':>8} | {'max|U-U*|':>9} {'conv':>5}")

    table = []
    for top_name in TOPS:
        for nx, ny, lx, ly in CASES:
            for name, res, err in run_case(nx, ny, lx, ly, top_name):
                print(f"{top_name:>5} {f'{nx}x{ny}':>9} {f'{lx:g}x{ly:g}':>7} | "
                      f"{name:>6} {res.iterations:>6} {res.tridiagonal_solves:>7} "
                      f"{res.seconds:>8.3f} | {err:>9.1e} {str(res.converged):>5}")
                table.append({
                    "top": top_name, "grid": f"{nx}x{ny}", "domain": f"{lx:g}x{ly:g}",
                    "method": name, "iterations": res.iterations,
                    "tridiagonal_solves": res.tridiagonal_solves,
                    "seconds": res.seconds, "max_err": err, "converged": res.converged,
                })
                # Эффективные методы обязаны сходиться к точному решению.
                if name in ("CR", "GMRES"):
                    assert res.converged and err < 1e-8, (top_name, nx, ny, name, err)
            print()
    return table


if __name__ == "__main__":
    run_laplace()
