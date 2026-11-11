"""
Опыт 5 ниши «подзадачи — нейросети»: 3D из той же одномерной сети.

Задача: −∇·(k ∇φ) = 0 в кубе [0,1]³, k(x, y, z) — гладкое поле, φ задана на гранях.
Три направления, три одномерных подсистемы:
    −(k u₁ₓ)ₓ = q₁,   −(k u₂ᵧ)ᵧ = q₂,   −(k u₃_z)_z = −q₁ − q₂,
u₁ = u₂ = u₃  ⇔  −∇·(k∇φ) = 0. Неизвестные — два координатора q₁, q₂.

Все одномерные решения делает сеть из опыта 4 (results/nn_line_hybrid.pt):
обучена ТОЛЬКО на одномерных линиях длиной 24–96, трёхмерных данных не видела.
Сеть линейна по правой части, поэтому координация — одна линейная система:
GMRES с грубым предобусловливателем (тот же оператор для постоянного k̄,
обращённый синус-преобразованием).

Эталон — та же координация с точной прогонкой (проверяется по семиточечной схеме).

    python nn_coordination_3d.py
"""

from __future__ import annotations

import math
import os
import time

import torch

import nn_coordination_grids as ng
from coordinator_laplace_descent import gmres


def to_last(t, axis):
    if axis == 2:
        return t
    if axis == 1:
        return t.transpose(1, 2)
    return t.permute(1, 2, 0)


def from_last(t, axis):
    if axis == 2:
        return t
    if axis == 1:
        return t.transpose(1, 2)
    return t.permute(2, 0, 1)


def problem3d(n_grid: int, seed: int, bc: str, contrast: float = 0.6):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(8, generator=g, dtype=torch.float64)

    def k(x, y, z):
        return torch.exp(contrast * (a[0] * torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.sin(math.pi * z)
                                     + a[1] * torch.cos(2 * math.pi * x) + a[2] * torch.cos(2 * math.pi * y)
                                     + a[3] * torch.cos(2 * math.pi * z)
                                     + 0.5 * a[4] * torch.sin(3 * math.pi * x + a[5]) * torch.cos(math.pi * z + a[6])))

    m = n_grid - 2
    h = 1.0 / (n_grid - 1)
    nd = torch.arange(1, m + 1, dtype=torch.float64) * h
    hf = (torch.arange(0, m + 1, dtype=torch.float64) + 0.5) * h
    Z, Y, X = nd[:, None, None], nd[None, :, None], nd[None, None, :]
    # k в полуузлах, уже в раскладке «направление линии — последняя ось».
    kx = k(hf[None, None, :], nd[None, :, None], nd[:, None, None])        # [z, y, x½]
    ky = k(nd[None, :, None], hf[None, None, :], nd[:, None, None])        # [z, x, y½]
    kz = k(nd[None, :, None], nd[:, None, None], hf[None, None, :])        # [y, x, z½]
    zero = torch.zeros(m, m, dtype=torch.float64)
    faces = {s: zero.clone() for s in ("x0", "x1", "y0", "y1", "z0", "z1")}
    if bc == "верх=1":
        faces["z1"] = torch.ones(m, m, dtype=torch.float64)               # [y, x]
    elif bc == "верх=sin·sin":
        faces["z1"] = torch.sin(math.pi * nd)[:, None] * torch.sin(math.pi * nd)[None, :]
    elif bc == "x=1 грань + низ горб":
        faces["x1"] = torch.ones(m, m, dtype=torch.float64)               # [z, y]
        faces["z0"] = (0.5 * (1 - torch.cos(2 * math.pi * nd)))[:, None] * torch.sin(math.pi * nd)[None, :]
    # Краевые значения каждой линии в той же раскладке, что и k.
    ends = {
        2: (faces["x0"], faces["x1"]),                                     # [z, y]
        1: (faces["y0"], faces["y1"]),                                     # [z, x]
        0: (faces["z0"], faces["z1"]),                                     # [y, x]
    }
    return dict(k={2: kx, 1: ky, 0: kz}, ends=ends, m=m, h=h,
                kbar=(kx.mean() + ky.mean() + kz.mean()).item() / 3)


def line_solve(solve, p, axis, f):
    m = p["m"]
    c = p["k"][axis].reshape(-1, m + 1)
    a, b = (t.reshape(-1) for t in p["ends"][axis])
    u = solve(c, to_last(f, axis).reshape(-1, m), a, b).reshape(m, m, m)
    return from_last(u, axis)


def residual(solve, p, q):
    u1 = line_solve(solve, p, 2, q[0])
    u2 = line_solve(solve, p, 1, q[1])
    u3 = line_solve(solve, p, 0, -q[0] - q[1])
    return torch.stack([u1 - u3, u2 - u3]), (u1 + u2 + u3) / 3


class Linear:
    """R(q) = R₀ + J q;  gmres решает J q = −R₀."""

    def __init__(self, solve, p) -> None:
        self.solve, self.p = solve, p
        self.r0, _ = residual(solve, p, torch.zeros(2, p["m"], p["m"], p["m"], dtype=torch.float64))
        self.tridiagonal_solves = 0

    def apply_m(self, d):
        return residual(self.solve, self.p, d)[0] - self.r0

    def residual(self, q):
        return -residual(self.solve, self.p, q)[0]


class Precond:
    """
    J⁻¹ для постоянного k̄, X = Y = Z = (k̄/h²)·T по своим осям:
    s = L⁻¹(X r₁ + Y r₂),  q₁ = X(r₁ − s),  q₂ = Y(r₂ − s),  L = X + Y + Z (через DST).
    """

    def __init__(self, p) -> None:
        m = p["m"]
        i = torch.arange(1, m + 1, dtype=torch.float64)
        self.s = torch.sin(math.pi * i[:, None] * i[None, :] / (m + 1)) * math.sqrt(2.0 / (m + 1))
        lam = 4 * torch.sin(i * math.pi / (2 * (m + 1))) ** 2
        self.c = p["kbar"] / p["h"] ** 2
        self.inv_l = 1.0 / (self.c * (lam[:, None, None] + lam[None, :, None] + lam[None, None, :]))

    def _t(self, w, axis):
        y = 2 * w
        y = y - torch.roll(w, 1, axis) - torch.roll(w, -1, axis)
        # roll переносит края — вернуть нулевые краевые условия
        idx0 = [slice(None)] * 3
        idx0[axis] = 0
        idxn = [slice(None)] * 3
        idxn[axis] = -1
        y[tuple(idx0)] += w.select(axis, -1)
        y[tuple(idxn)] += w.select(axis, 0)
        return self.c * y

    def _dst3(self, w):
        # По одной оси за раз (einsum сразу по трём осям строит огромный промежуточный массив).
        w = torch.tensordot(self.s, w, dims=([1], [0]))
        w = torch.tensordot(w, self.s, dims=([1], [1])).permute(0, 2, 1)
        return torch.tensordot(w, self.s, dims=([2], [1]))

    def __call__(self, r):
        r1, r2 = r[0], r[1]
        rhs = self._t(r1, 2) + self._t(r2, 1)
        s = self._dst3(self.inv_l * self._dst3(rhs))
        return torch.stack([self._t(r1 - s, 2), self._t(r2 - s, 1)])


def coordinate(solve, p, rtol):
    lin = Linear(solve, p)
    q0 = torch.zeros(2, p["m"], p["m"], p["m"], dtype=torch.float64)
    res = gmres(lin, q0, rtol=rtol, max_iter=300, restart=150, preconditioner=Precond(p))
    r, phi = residual(solve, p, res.q)
    return phi, res.iterations, (torch.linalg.vector_norm(r) / torch.linalg.vector_norm(lin.r0)).item()


def scheme_residual(p, phi):
    """Невязка семиточечной схемы Σ_оси (A φ − краевые) относительно краевых членов."""
    total = torch.zeros_like(phi)
    bnorm = 0.0
    m = p["m"]
    for axis in (0, 1, 2):
        c = p["k"][axis].reshape(-1, m + 1)
        a, b = (t.reshape(-1) for t in p["ends"][axis])
        w = to_last(phi, axis).reshape(-1, m)
        aw = ng.apply_a(c, w)
        aw[:, 0] -= c[:, 0] * a
        aw[:, -1] -= c[:, -1] * b
        total += from_last(aw.reshape(m, m, m), axis)
        bnorm += (c[:, 0] * a).norm().item() + (c[:, -1] * b).norm().item()
    return total.norm().item() / bnorm


def main() -> None:
    if os.environ.get("LINE") == "refined":      # LINE=refined: сеть v3 с уточнением (nn_line_refined.py)
        import nn_line_refined
        net = nn_line_refined.load()
        print(f"Сеть v3 из {nn_line_refined.WEIGHTS}, уточнений R = {net.refine}")
    else:
        net = ng.LineHybrid()
        net.load_state_dict(torch.load(ng.WEIGHTS))
        net = net.double().eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        print(f"Сеть: {ng.WEIGHTS} (обучена на 1D-линиях длиной {ng.TRAIN_N[0]}–{ng.TRAIN_N[1]}, без дообучения)")
    print(f"{'сетка':>8} | {'неизв.':>9} | {'seed':>4} | {'граничные условия':22} | {'ошибка φ':>8} | "
          f"{'итер.':>5} | {'сеть':>7} | {'прогонка':>8} | {'невязка схемы эталона':>12}")
    for n_grid in (26, 50, 98):
        errs = []
        for seed in (31, 32):
            for bc in ("верх=1", "верх=sin·sin", "x=1 грань + низ горб"):
                p = problem3d(n_grid, seed, bc)
                t0 = time.perf_counter()
                phi_ref, it_ref, _ = coordinate(ng.exact_line, p, 1e-11)
                t_ref = time.perf_counter() - t0
                check = scheme_residual(p, phi_ref)
                t0 = time.perf_counter()
                phi, it, _ = coordinate(net, p, 1e-9)
                t_nn = time.perf_counter() - t0
                err = (torch.linalg.vector_norm(phi - phi_ref) / torch.linalg.vector_norm(phi_ref)).item()
                errs.append(err)
                print(f"{n_grid - 2:>3}³     | {(n_grid - 2) ** 3:>9,} | {seed:>4} | {bc:22} | {err:8.1e} | {it:5d} | "
                      f"{t_nn:6.1f}с | {t_ref:7.1f}с | {check:.0e} (итер. {it_ref})", flush=True)
        print(f"{'':>8}   средняя {sum(errs) / len(errs):.2e}, худшая {max(errs):.2e}\n", flush=True)


if __name__ == "__main__":
    main()
