"""
Экологическая модель (eco_pollution.py), переменные K, w, λ:
  1. координация совпадает с прямым решением двумерной системы (плотная матрица);
  2. концентрация неотрицательна (схема против потока монотонна);
  3. J монотонно убывает;
  4. вклады источников складываются в полное решение;
  5. сопряжённая задача: c(рецептор) = ⟨φ, S⟩;
  6. итераций с ADI мало и почти не растёт при измельчении сетки.
"""

from common import DTYPE

import torch

from eco_pollution import EcoSolver, Region, build_operators, dense_operator, nearest_node


def _monotone(history):
    return all(b <= a * (1 + 1e-12) for a, b in zip(history, history[1:]))


def test_eco() -> None:
    region = Region(nx=41, ny=21)
    ops = build_operators(region, dtype=DTYPE)
    solver = EcoSolver(ops)
    c, res = solver.solve(ops.source, rtol=1e-12)

    direct = torch.linalg.solve(dense_operator(ops), ops.source.reshape(-1)).reshape(c.shape)
    err = (torch.max(torch.abs(c - direct)) / torch.max(direct)).item()
    assert res.converged and err < 1e-9, err
    assert c.min().item() > -1e-12 and direct.min().item() >= 0.0
    assert _monotone(res.history)

    parts = [solver.solve(ops.interior_source([s]), rtol=1e-12)[0] for s in region.sources]
    sup = torch.max(torch.abs(sum(parts) - c)).item()
    assert sup < 1e-9 * c.max().item(), sup

    cell = region.hx * region.hy
    ri, rj = nearest_node(region, 12.0, 6.0)
    delta = torch.zeros_like(c)
    delta[ri, rj] = 1.0 / cell
    phi, _ = EcoSolver(ops, adjoint=True).solve(delta, rtol=1e-12)
    adj = (torch.sum(phi * ops.source) * cell).item()
    assert abs(adj - c[ri, rj].item()) < 1e-9 * abs(c[ri, rj].item()), (adj, c[ri, rj].item())

    iters = []
    for nx, ny in ((101, 51), (201, 101), (401, 201)):
        big = build_operators(Region(nx=nx, ny=ny), dtype=DTYPE)
        _, r = EcoSolver(big).solve(big.source)
        assert r.converged
        iters.append(r.iterations)
    assert max(iters) <= 15, iters

    print(f"test_eco 41x21: {res.iterations} it, rel err vs direct {err:.1e}, "
          f"superposition {sup:.1e}, adjoint {adj:.6f} = {c[ri, rj].item():.6f}; "
          f"ADI iterations on 101x51 / 201x101 / 401x201: {iters}")
    print("test_eco: OK")


if __name__ == "__main__":
    test_eco()
