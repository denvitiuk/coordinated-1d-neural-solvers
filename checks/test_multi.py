"""
Общий метод координации K подсистем (multi_coordination.py):
  1. K = 3 случайные плотные подсистемы: градиент против autograd от J,
     записанного напрямую как ½ Σ_{i<j} ‖z_i - z_j‖²; CGLS -> q* из lstsq.
  2. K = 2 из одномерных прогонок воспроизводит пластину диплома.
  3. «Варшава — Киев»: две перекрывающиеся области, каждая со своим решателем,
     согласуются в общей зоне и дают решение всей области.
"""

import itertools

from common import DTYPE

import torch

from multi_coordination import (
    MultiCoordinationProblem,
    Operator,
    OverlapDemo,
    Subsystem,
    coordinate_many,
)
from plate_coordination import Plate, build_plate_problem, exact_plate


def _monotone(history):
    return all(b <= a * (1 + 1e-12) for a, b in zip(history, history[1:]))


def test_random_three() -> None:
    gen = torch.Generator().manual_seed(5)
    n, m, k = 8, 6, 3  # u_k ∈ R^8, q ∈ R^6, три подсистемы
    mats, ds, fs = [], [], []
    subs = []
    for _ in range(k):
        a = torch.randn((n, n), generator=gen, dtype=DTYPE) + 6.0 * torch.eye(n, dtype=DTYPE)
        d = torch.randn((n, m), generator=gen, dtype=DTYPE)
        f = torch.randn(n, generator=gen, dtype=DTYPE)
        mats.append(a); ds.append(d); fs.append(f)
        subs.append(Subsystem.dense(a, Operator.matrix(d, (m,), (n,)), f))
    problem = MultiCoordinationProblem(subs)

    q = torch.randn(m, generator=gen, dtype=DTYPE)
    g_adj = -problem.apply_m_t(problem.residual(q))

    q_ag = q.clone().requires_grad_(True)
    us = [torch.linalg.solve(a, d @ q_ag + f) for a, d, f in zip(mats, ds, fs)]
    j = 0.5 * sum(torch.sum((us[i] - us[l]) ** 2) for i, l in itertools.combinations(range(k), 2))
    j.backward()
    grad_err = (torch.linalg.vector_norm(g_adj - q_ag.grad) / torch.linalg.vector_norm(q_ag.grad)).item()

    j_resid = 0.5 * torch.sum(problem.residual(q) ** 2).item()
    assert abs(j_resid - j.item()) < 1e-10 * max(1.0, j.item()), "residual does not match J"

    # Эталон: lstsq по всем парам разностей.
    rows, rhs = [], []
    for i, l in itertools.combinations(range(k), 2):
        rows.append(torch.linalg.solve(mats[i], ds[i]) - torch.linalg.solve(mats[l], ds[l]))
        rhs.append(torch.linalg.solve(mats[l], fs[l]) - torch.linalg.solve(mats[i], fs[i]))
    q_star = torch.linalg.lstsq(torch.cat(rows), torch.cat(rhs).unsqueeze(1)).solution.squeeze(1)

    res = coordinate_many(problem, torch.zeros(m, dtype=DTYPE), rtol=1e-12)
    err = torch.linalg.vector_norm(res.q - q_star).item()
    print(f"test_multi K=3: grad rel err {grad_err:.1e}, CGLS {res.iterations} it, ||q - q*|| {err:.1e}")
    assert grad_err < 1e-10 and res.converged and err < 1e-8 and _monotone(res.history)


def test_plate_as_two_subsystems() -> None:
    plate = Plate(mx=16, ny=36)
    pr = build_plate_problem(plate, "cpu", DTYPE)
    hx2, hy2 = pr.dx.scale, pr.dy.scale

    rows = Subsystem(
        solve=pr._a.solve,
        solve_t=pr._a_t.solve,
        d=Operator(lambda q: -hx2 * q, lambda y: -hx2 * y),
        f=pr.fx,
        name="rows (x)",
    )
    cols = Subsystem(
        solve=lambda b: pr._b.solve(b.T).T,
        solve_t=lambda b: pr._b_t.solve(b.T).T,
        d=Operator(lambda q: hy2 * q, lambda y: hy2 * y),
        f=pr.fy,
        name="columns (y)",
    )
    problem = MultiCoordinationProblem([rows, cols])
    res = coordinate_many(problem, torch.zeros_like(pr.fx), rtol=1e-12, max_iter=20_000)

    t = torch.zeros((plate.ny, plate.mx), dtype=DTYPE)
    t[1:-1, 1:-1] = problem.consensus(res.q)
    t[1:-1, 0], t[1:-1, -1] = t[1:-1, 1], t[1:-1, -2]
    exact = exact_plate(plate)
    t[0], t[-1] = exact[0], exact[-1]
    err = torch.max(torch.abs(t - exact)).item()
    print(f"test_multi plate as 2 black boxes: CGLS {res.iterations} it, max|T - T*| {err:.1e}")
    assert res.converged and err < 1e-7


def test_warsaw_kyiv() -> None:
    demo = OverlapDemo()
    problem = demo.build()
    res = coordinate_many(problem, torch.zeros(demo.n_q, dtype=DTYPE), rtol=1e-12)

    u_w, u_k = problem.states(res.q)
    glob = demo.global_solution()
    err_w = torch.max(torch.abs(u_w - glob[:, [c - 1 for c in demo.cols1]])).item()
    err_k = torch.max(torch.abs(u_k - glob[:, [c - 1 for c in demo.cols2]])).item()
    mismatch = torch.max(torch.abs(problem.residual(res.q))).item()
    print(f"test_multi Warsaw-Kyiv overlap: CGLS {res.iterations} it, "
          f"overlap mismatch {mismatch:.1e}, vs global: Warsaw {err_w:.1e}, Kyiv {err_k:.1e}")
    assert res.converged and _monotone(res.history)
    assert err_w < 1e-8 and err_k < 1e-8
    print("test_multi: OK")


if __name__ == "__main__":
    test_random_three()
    test_plate_as_two_subsystems()
    test_warsaw_kyiv()
