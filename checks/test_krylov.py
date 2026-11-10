"""
Разложение прогонки, CR и GMRES:
  - TridiagonalFactor.solve == solve_batched_tridiagonal;
  - GMRES на несимметричной задаче → q*, J монотонна, дешевле CGLS по прогонкам;
  - GMRES с маленьким restart тоже сходится и монотонен;
  - CR на симметричной (лапласовой) задаче совпадает с точным решением;
  - CR отказывается работать с несимметричной M.
"""

from common import DTYPE, build_general_problem, exact_minimizer

import torch

from batched_tridiagonal import TridiagonalFactor, solve_batched_tridiagonal
from coordinator_laplace_descent import (
    assemble_solution,
    build_laplace_problem,
    cgls,
    conjugate_residual,
    coordinate,
    gmres,
)
from laplace_exact import exact_discrete_solution

RTOL = 1e-12


def _monotone(history: list[float]) -> bool:
    return all(b <= a * (1 + 1e-12) for a, b in zip(history, history[1:]))


def test_factor_matches_thomas() -> None:
    gen = torch.Generator().manual_seed(3)
    lo = torch.randn((4, 9), generator=gen, dtype=DTYPE)
    up = torch.randn((4, 9), generator=gen, dtype=DTYPE)
    di = 6.0 + torch.rand((4, 9), generator=gen, dtype=DTYPE)
    lo[:, 0] = 0
    up[:, -1] = 0
    rhs = torch.randn((4, 9), generator=gen, dtype=DTYPE)

    reference = solve_batched_tridiagonal(lo, di, up, rhs)
    errs = {}
    for method in ("thomas", "dense"):
        # разные коэффициенты в каждой системе batch
        errs[method] = torch.max(torch.abs(
            TridiagonalFactor(lo, di, up, method=method).solve(rhs) - reference)).item()
        # общие коэффициенты для всего batch (путь «одна матрица»)
        shared = [t[:1].expand_as(t).clone() for t in (lo, di, up)]
        errs[method + "-shared"] = torch.max(torch.abs(
            TridiagonalFactor(*shared, method=method).solve(rhs)
            - solve_batched_tridiagonal(*shared, rhs))).item()
    assert max(errs.values()) < 1e-13, errs
    print("test_factor_matches_thomas: " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()) + " OK")


def test_gmres_general() -> None:
    problem = build_general_problem(seed=0)
    q0 = torch.zeros_like(problem.fx)
    q_star = exact_minimizer(problem)

    assert not problem.m_is_symmetric()
    res = gmres(problem, q0, rtol=RTOL)
    ref = cgls(problem, q0, rtol=RTOL)
    small = gmres(problem, q0, rtol=RTOL, restart=5, max_iter=5000)
    auto = coordinate(problem, q0, rtol=RTOL)

    for r in (res, small, auto):
        err = torch.linalg.vector_norm(r.q - q_star).item()
        assert r.converged and err < 1e-6, (r.method, err)
        assert _monotone(r.history), r.method

    print(
        f"test_gmres_general: GMRES {res.iterations} it / {res.tridiagonal_solves} solves, "
        f"GMRES(5) {small.iterations} it, CGLS {ref.iterations} it / "
        f"{ref.tridiagonal_solves} solves, auto -> {auto.method}"
    )
    assert auto.method == "GMRES"
    assert res.tridiagonal_solves < ref.tridiagonal_solves
    print("test_gmres_general: OK")


def test_cr_symmetric() -> None:
    nx, ny, lx, ly = 23, 15, 2.0, 1.0
    top = lambda x, lx: torch.ones_like(x)  # все гармоники, не одна синусоида
    problem = build_laplace_problem(nx, ny, "cpu", DTYPE, lx=lx, ly=ly, top=top)
    q0 = torch.zeros_like(problem.fx)
    exact = exact_discrete_solution(nx, ny, lx, ly, top=top)

    assert problem.m_is_symmetric()
    res = conjugate_residual(problem, q0, rtol=RTOL)
    err = torch.max(torch.abs(assemble_solution(problem, res.q, lx, top) - exact)).item()
    assert res.converged and err < 1e-9, err
    assert _monotone(res.history)
    assert coordinate(problem, q0, rtol=RTOL).method == "GMRES+P"

    try:
        conjugate_residual(build_general_problem(), torch.zeros_like(problem.fx))
    except ValueError:
        pass
    else:
        raise AssertionError("CR accepted a non-symmetric M")

    print(f"test_cr_symmetric: {res.iterations} iterations, max|U - U*| = {err:.1e} OK")


if __name__ == "__main__":
    test_factor_matches_thomas()
    test_gmres_general()
    test_cr_symmetric()
