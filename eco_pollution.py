"""
Экологическая задача: стационарное распространение загрязнения в воздухе над районом.

    -∇·(K ∇c) + w·∇c + λ c = S        в прямоугольнике [0, Lx] x [0, Ly] (км)
    c = 0                              на границе (фон вычтен)

c(x, y)  — концентрация (условные единицы),
K(x, y)  — турбулентная диффузия, км²/ч (над городом больше — шероховатость),
w(x, y)  — ветер, км/ч (в основном западный, со сдвигом и меандром),
λ(x, y)  — выведение, 1/ч (химический распад + осаждение; над лесом больше),
S(x, y)  — источники: ТЭЦ, промзона, автобан (линейный источник).

Все коэффициенты переменные, поэтому разложение по собственным векторам и БПФ
(точный метод для пластины) здесь не работают. Метод координации работает так же:

    L = Lx + Ly,   Lx = -∂x(K∂x) + wx∂x + λ/2,   Ly = -∂y(K∂y) + wy∂y + λ/2,
    Lx u = -q + S/2   (прогонки вдоль x),   Ly v = q + S/2   (прогонки вдоль y),
    J(q) = ½‖u - v‖² -> 0,  тогда c = u = v и (Lx + Ly) c = S.

Схема: консервативная диффузия (K на гранях ячеек), адвекция против потока
(монотонна: c ≥ 0 при S ≥ 0, без осцилляций). Обе матрицы — M-матрицы.

Модель демонстрационная: параметры правдоподобного порядка, но не откалиброваны
по реальным измерениям.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from coordinator_laplace_descent import (
    AdiPreconditioner,
    CoordinatorProblem,
    LinearMap,
    SolveResult,
    gmres,
    tridiagonal_matvec,
)
from batched_tridiagonal import transpose_tridiagonal

Tensor = torch.Tensor


@dataclass
class Source:
    name: str
    x: float  # км
    y: float
    rate: float  # интенсивность, усл. ед./ч
    sigma: float = 0.3  # размер пятна, км
    # Линейный источник: отрезок (x, y) -> (x_end, y_end).
    x_end: float | None = None
    y_end: float | None = None


def default_sources() -> list[Source]:
    return [
        Source("ТЭЦ", 4.0, 6.5, rate=100.0, sigma=0.25),
        Source("Промзона", 9.5, 3.0, rate=60.0, sigma=0.8),
        Source("Автобан", 2.0, 1.0, rate=40.0, sigma=0.2, x_end=18.0, y_end=8.5),
    ]


@dataclass
class Region:
    lx: float = 20.0  # км, запад -> восток
    ly: float = 10.0  # км, юг -> север
    nx: int = 201  # узлов вместе с границей
    ny: int = 101
    # Диффузия: фон + добавка над городом.
    k0: float = 1.5
    k_city: float = 4.0
    city: tuple[float, float, float] = (8.0, 5.0, 2.5)  # центр x, y, радиус
    # Ветер: западный с вертикальным сдвигом и меандром.
    wind: float = 8.0
    meander: float = 1.5
    # Выведение: фон + лес.
    decay: float = 0.05
    forest_decay: float = 0.6
    forest: tuple[float, float, float, float] = (13.0, 18.0, 1.0, 4.5)  # x0, x1, y0, y1
    sources: list[Source] = field(default_factory=default_sources)

    @property
    def hx(self) -> float:
        return self.lx / (self.nx - 1)

    @property
    def hy(self) -> float:
        return self.ly / (self.ny - 1)

    def grid(self, dtype=torch.float64) -> tuple[Tensor, Tensor]:
        x = torch.linspace(0.0, self.lx, self.nx, dtype=dtype)
        y = torch.linspace(0.0, self.ly, self.ny, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return xx, yy  # [ny, nx]

    # --- поля коэффициентов на всей сетке [ny, nx] ---------------------------

    def diffusion(self, xx: Tensor, yy: Tensor) -> Tensor:
        cx, cy, r = self.city
        return self.k0 + self.k_city * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / r**2)

    def velocity(self, xx: Tensor, yy: Tensor) -> tuple[Tensor, Tensor]:
        wx = self.wind * (0.6 + 0.4 * yy / self.ly)
        wy = self.meander * torch.sin(2 * math.pi * xx / self.lx)
        return wx, wy

    def removal(self, xx: Tensor, yy: Tensor) -> Tensor:
        x0, x1, y0, y1 = self.forest
        # Мягкая маска леса (ширина перехода 0.3 км).
        mask = (torch.sigmoid((xx - x0) / 0.3) * torch.sigmoid((x1 - xx) / 0.3)
                * torch.sigmoid((yy - y0) / 0.3) * torch.sigmoid((y1 - yy) / 0.3))
        return self.decay + self.forest_decay * mask

    def source_field(self, xx: Tensor, yy: Tensor, sources: list[Source] | None = None) -> Tensor:
        """Сумма источников; каждый нормирован так, что ∫S = rate."""
        total = torch.zeros_like(xx)
        for s in self.sources if sources is None else sources:
            if s.x_end is None:
                d2 = (xx - s.x) ** 2 + (yy - s.y) ** 2
                length = None
            else:
                ex, ey = s.x_end - s.x, s.y_end - s.y
                length = math.hypot(ex, ey)
                t = (((xx - s.x) * ex + (yy - s.y) * ey) / length**2).clamp(0.0, 1.0)
                d2 = (xx - s.x - t * ex) ** 2 + (yy - s.y - t * ey) ** 2
            blob = torch.exp(-d2 / (2 * s.sigma**2))
            norm = torch.sum(blob) * self.hx * self.hy
            total += s.rate * blob / norm
        return total


# ---------------------------------------------------------------------------
# Дискретизация
# ---------------------------------------------------------------------------


def _direction(k_minus: Tensor, k_plus: Tensor, w: Tensor, react: Tensor, h: float):
    """
    Одномерный оператор -(K c')' + w c' + react·c, пакет систем [batch, n].
    K на гранях j-½ и j+½, адвекция против потока.
    """
    wp, wm = w.clamp(min=0.0), w.clamp(max=0.0)
    lower = -k_minus / h**2 - wp / h
    upper = -k_plus / h**2 + wm / h
    diagonal = (k_minus + k_plus) / h**2 + (wp - wm) / h + react
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    return lower.contiguous(), diagonal.contiguous(), upper.contiguous()


@dataclass
class EcoOperators:
    region: Region
    a: tuple  # Lx: [ny_in, nx_in], системы вдоль x
    b: tuple  # Ly: [nx_in, ny_in], системы вдоль y
    source: Tensor  # S во внутренних узлах [ny_in, nx_in]
    xx: Tensor
    yy: Tensor

    def apply_l(self, c: Tensor) -> Tensor:
        """Полный двумерный оператор L c = Lx c + Ly c (для проверки)."""
        return tridiagonal_matvec(self.a, c) + tridiagonal_matvec(self.b, c.T).T

    def apply_l_t(self, c: Tensor) -> Tensor:
        return (tridiagonal_matvec(transpose_tridiagonal(*self.a), c)
                + tridiagonal_matvec(transpose_tridiagonal(*self.b), c.T).T)

    def interior_source(self, sources: list[Source] | None) -> Tensor:
        return self.region.source_field(self.xx, self.yy, sources)[1:-1, 1:-1]


def build_operators(region: Region, device="cpu", dtype=torch.float64) -> EcoOperators:
    xx, yy = region.grid(torch.float64)
    k = region.diffusion(xx, yy)
    wx, wy = region.velocity(xx, yy)
    lam = region.removal(xx, yy)
    inner = (slice(1, -1), slice(1, -1))

    # Вдоль x: batch = строки (y), система вдоль x.
    kx_minus = 0.5 * (k[1:-1, :-2] + k[1:-1, 1:-1])
    kx_plus = 0.5 * (k[1:-1, 1:-1] + k[1:-1, 2:])
    a = _direction(kx_minus, kx_plus, wx[inner], 0.5 * lam[inner], region.hx)

    # Вдоль y: batch = столбцы (x), система вдоль y -> транспонированные массивы.
    ky_minus = 0.5 * (k[:-2, 1:-1] + k[1:-1, 1:-1])
    ky_plus = 0.5 * (k[1:-1, 1:-1] + k[2:, 1:-1])
    b = _direction(ky_minus.T, ky_plus.T, wy[inner].T, 0.5 * lam[inner].T, region.hy)

    to = dict(device=device, dtype=dtype)
    source = region.source_field(xx, yy)[inner]
    return EcoOperators(
        region,
        tuple(t.to(**to) for t in a),
        tuple(t.to(**to) for t in b),
        source.to(**to),
        xx.to(**to),
        yy.to(**to),
    )


# ---------------------------------------------------------------------------
# Решение координацией
# ---------------------------------------------------------------------------


class EcoSolver:
    """
    Разложения прогонки и ADI строятся один раз; затем любое число решений
    с разными источниками (сценарии, вклад каждого источника) и сопряжённые
    задачи (карта влияния на точку-рецептор).
    """

    def __init__(self, ops: EcoOperators, adi_steps: int = 6, solver: str = "auto",
                 adjoint: bool = False) -> None:
        self.ops = ops
        a, b = ops.a, ops.b
        if adjoint:
            a, b = transpose_tridiagonal(*a), transpose_tridiagonal(*b)
        zero = torch.zeros_like(ops.source)
        one = LinearMap(scale=1.0)
        self.problem = CoordinatorProblem(a, b, one, one, zero, zero.clone(), solver=solver)
        self.pre = AdiPreconditioner(self.problem, steps=adi_steps) if adi_steps else None

    def solve(self, rhs: Tensor, rtol: float = 1e-10, max_iter: int = 10_000,
              log_every: int = 0) -> tuple[Tensor, SolveResult]:
        self.problem.fx = 0.5 * rhs
        self.problem.fy = 0.5 * rhs
        q0 = torch.zeros_like(rhs)
        res = gmres(self.problem, q0, rtol=rtol, max_iter=max_iter,
                    log_every=log_every, preconditioner=self.pre)
        u, v = self.problem.states(res.q)
        return 0.5 * (u + v), res


def full_field(c_inner: Tensor) -> Tensor:
    """Добавить нулевую границу: [ny, nx]."""
    ny_in, nx_in = c_inner.shape
    out = torch.zeros((ny_in + 2, nx_in + 2), dtype=c_inner.dtype, device=c_inner.device)
    out[1:-1, 1:-1] = c_inner
    return out


def dense_operator(ops: EcoOperators) -> Tensor:
    """Двумерная матрица L [N, N] (только для проверки на малых сетках)."""
    (al, ad, au), (bl, bd, bu) = (tuple(t.cpu().double() for t in ops.a),
                                  tuple(t.cpu().double() for t in ops.b))
    ny_in, nx_in = ad.shape
    n = ny_in * nx_in
    idx = torch.arange(n).reshape(ny_in, nx_in)
    mat = torch.zeros((n, n), dtype=torch.float64)
    mat[idx.reshape(-1), idx.reshape(-1)] = (ad + bd.T).reshape(-1)
    mat[idx[:, 1:].reshape(-1), idx[:, :-1].reshape(-1)] = al[:, 1:].reshape(-1)
    mat[idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)] = au[:, :-1].reshape(-1)
    mat[idx[1:, :].reshape(-1), idx[:-1, :].reshape(-1)] = bl.T[1:, :].reshape(-1)
    mat[idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1)] = bu.T[:-1, :].reshape(-1)
    return mat


def nearest_node(region: Region, x: float, y: float) -> tuple[int, int]:
    """Индекс внутреннего узла [i, j], ближайшего к точке (x, y)."""
    j = min(max(round(x / region.hx) - 1, 0), region.nx - 3)
    i = min(max(round(y / region.hy) - 1, 0), region.ny - 3)
    return i, j


# ---------------------------------------------------------------------------
# Отчётный расчёт
# ---------------------------------------------------------------------------

RECEPTOR = ("Больница", 12.0, 6.0)
LIMIT = 1.0  # порог для «зоны превышения», усл. ед.


def main() -> None:
    import numpy as np

    from rectangle_report import heatmap, write_png

    region = Region()
    start = time.perf_counter()
    ops = build_operators(region)
    solver = EcoSolver(ops)
    setup = time.perf_counter() - start

    c, res = solver.solve(ops.source, log_every=0)
    resid = (torch.linalg.vector_norm(ops.apply_l(c) - ops.source)
             / torch.linalg.vector_norm(ops.source)).item()

    print(f"=== Загрязнение воздуха, район {region.lx:g} x {region.ly:g} км, "
          f"сетка {region.nx} x {region.ny} ({c.numel()} неизвестных) ===")
    print(f"Подготовка (разложения + ADI): {setup:.3f} с")
    print(f"{res.method}: {res.iterations} итераций, {res.tridiagonal_solves} пакетов прогонок, "
          f"{res.seconds:.3f} с, ‖Lc − S‖/‖S‖ = {resid:.1e}, min c = {c.min().item():.1e}")

    cell = region.hx * region.hy
    i_max = int(torch.argmax(c))
    iy, jx = divmod(i_max, c.shape[1])
    print(f"Максимум c = {c.max().item():.3f} в точке x = {(jx + 1) * region.hx:.1f}, "
          f"y = {(iy + 1) * region.hy:.1f} км")
    print(f"Площадь с c > {LIMIT}: {(c > LIMIT).sum().item() * cell:.1f} км²")

    # Вклад каждого источника: задача линейна, c = Σ c_k.
    print("\nВклад источников (то же разложение, только новые правые части):")
    name, rx, ry = RECEPTOR
    ri, rj = nearest_node(region, rx, ry)
    parts = {}
    for s in region.sources:
        ck, rk = solver.solve(ops.interior_source([s]))
        parts[s.name] = ck
        print(f"  {s.name:9s}: {rk.iterations} it, {rk.seconds:.3f} с, "
              f"в точке «{name}» {ck[ri, rj].item():.4f}, "
              f"площадь c > {LIMIT}: {(ck > LIMIT).sum().item() * cell:.1f} км²")
    total = sum(parts.values())
    print(f"  сумма вкладов − полное решение: {torch.max(torch.abs(total - c)).item():.1e}")

    # Сценарий без пересчёта: фильтр на ТЭЦ −50 %.
    scen = c - 0.5 * parts["ТЭЦ"]
    print(f"\nСценарий «фильтр на ТЭЦ, −50 %» (по линейности, без нового решения): "
          f"«{name}» {c[ri, rj].item():.4f} -> {scen[ri, rj].item():.4f}, "
          f"площадь c > {LIMIT}: {(c > LIMIT).sum().item() * cell:.1f} -> "
          f"{(scen > LIMIT).sum().item() * cell:.1f} км²")

    # Сопряжённая задача: карта влияния на рецептор, Lᵀφ = δ.
    adj = EcoSolver(ops, adjoint=True)
    delta = torch.zeros_like(c)
    delta[ri, rj] = 1.0 / cell
    phi, ra = adj.solve(delta)
    via_adjoint = (torch.sum(phi * ops.source) * cell).item()
    print(f"\nКарта влияния на «{name}» (одна сопряжённая задача, {ra.iterations} it): "
          f"c_рецептор = ⟨φ, S⟩ = {via_adjoint:.4f} (прямой расчёт {c[ri, rj].item():.4f})")

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    tag = f"{region.nx}x{region.ny}"
    np.save(out / f"eco_concentration_{tag}.npy", full_field(c).numpy())
    # √c для картинки: иначе виден только пик у ТЭЦ.
    write_png(out / f"eco_concentration_{tag}.png", heatmap(np.sqrt(full_field(c).numpy())[::-1]), 4)
    write_png(out / f"eco_influence_{tag}.png", heatmap(full_field(phi).numpy()[::-1]), 4)
    print(f"\nСохранено: results/eco_concentration_{tag}.png/.npy, results/eco_influence_{tag}.png")


if __name__ == "__main__":
    main()
