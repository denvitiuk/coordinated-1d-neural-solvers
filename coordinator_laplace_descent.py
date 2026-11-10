"""
Метод функции-координатора.

    A u = -DX q + FX        (системы вдоль строк q)
    B v =  DY q + FY        (системы вдоль столбцов q)

    r(q) = u - v = c - M q,     M = A⁻¹DX + B⁻¹DY
    J(q) = ½‖r‖²
    ∇J   = -(DXᵀp + DYᵀs),      Aᵀp = r,  Bᵀs = r

Методы (все минимизируют ‖r‖ и проверяют монотонность J):
  steepest_descent  — точный шаг; скорость ∝ κ(M)²; 4 прогонки/итер.
  cgls              — CG на нормальных уравнениях; ∝ κ(M); 4 прогонки/итер.
  conjugate_residual— M = Mᵀ > 0; ∝ √κ(M); 2 прогонки/итер.
  gmres             — любая невырожденная M; ∝ √κ (для хорошего спектра);
                      2 прогонки/итер. + ортогонализация.
  coordinate        — выбирает CR или GMRES автоматически.

Все величины — 2D-сетки [rows, cols]:
  A: batch = rows, система вдоль cols, коэффициенты [rows, cols];
  B: batch = cols, система вдоль rows, коэффициенты [cols, rows];
  DX, DY: линейные отображения сетки в сетку (LinearMap).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from batched_tridiagonal import (
    TridiagonalFactor,
    get_device,
    synchronize,
    transpose_tridiagonal,
)

Tridiagonal = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Операторы
# ---------------------------------------------------------------------------


@dataclass
class LinearMap:
    """
    Линейное отображение сетки q [rows, cols] в сетку того же размера.

    Ровно одно из полей задано:
      scale    — s·I;
      diagonal — поэлементное умножение на сетку [rows, cols];
      dense    — плотная матрица [N, N] над q.reshape(-1), N = rows·cols.
    """

    scale: float | None = None
    diagonal: torch.Tensor | None = None
    dense: torch.Tensor | None = None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is not None:
            return self.scale * x
        if self.diagonal is not None:
            return self.diagonal * x
        return (self.dense @ x.reshape(-1)).reshape(x.shape)

    def apply_t(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is not None:
            return self.scale * x
        if self.diagonal is not None:
            return self.diagonal * x
        return (self.dense.T @ x.reshape(-1)).reshape(x.shape)


def _is_symmetric(coefficients: Tridiagonal) -> bool:
    lower, _, upper = coefficients
    return bool(torch.equal(lower[:, 1:], upper[:, :-1]))


@dataclass
class CoordinatorProblem:
    a: Tridiagonal  # [rows, cols]
    b: Tridiagonal  # [cols, rows]
    dx: LinearMap
    dy: LinearMap
    fx: torch.Tensor  # [rows, cols]
    fy: torch.Tensor  # [rows, cols] (в ориентации q, не транспонировано)
    # Способ прогонки: "auto" | "dense" (треугольные BLAS) | "thomas" (цикл) | "pcr" (GPU).
    solver: str = "auto"
    # Число решённых пакетов трёхдиагональных систем (одна прогонка по
    # всем строкам или всем столбцам = 1).
    tridiagonal_solves: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # Разложение для прогонки — один раз на всю оптимизацию.
        m = self.solver
        self._a = TridiagonalFactor(*self.a, method=m)
        self._b = TridiagonalFactor(*self.b, method=m)
        self._a_t = TridiagonalFactor(*transpose_tridiagonal(*self.a), method=m)
        self._b_t = TridiagonalFactor(*transpose_tridiagonal(*self.b), method=m)

    # --- прогонки ---------------------------------------------------------

    def _rows(self, factor: TridiagonalFactor, rhs: torch.Tensor) -> torch.Tensor:
        self.tridiagonal_solves += 1
        return factor.solve(rhs)

    def _columns(self, factor: TridiagonalFactor, rhs: torch.Tensor) -> torch.Tensor:
        self.tridiagonal_solves += 1
        return factor.solve(rhs.T).T

    # --- операторы задачи -------------------------------------------------

    def states(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Прямые задачи: u из A, v из B."""
        u = self._rows(self._a, -self.dx.apply(q) + self.fx)
        v = self._columns(self._b, self.dy.apply(q) + self.fy)
        return u, v

    def residual(self, q: torch.Tensor) -> torch.Tensor:
        u, v = self.states(q)
        return u - v

    def apply_m(self, g: torch.Tensor) -> torch.Tensor:
        """M g = A⁻¹DX g + B⁻¹DY g."""
        return self._rows(self._a, self.dx.apply(g)) + self._columns(
            self._b, self.dy.apply(g)
        )

    def apply_m_t(self, y: torch.Tensor) -> torch.Tensor:
        """Mᵀ y = DXᵀA⁻ᵀ y + DYᵀB⁻ᵀ y."""
        p = self._rows(self._a_t, y)
        s = self._columns(self._b_t, y)
        return self.dx.apply_t(p) + self.dy.apply_t(s)

    def m_is_symmetric(self) -> bool:
        """
        Достаточное условие M = Mᵀ: A, B симметричны, DX, DY = s·I.
        (A⁻¹·sI симметрична, если A симметрична.)
        """
        return (
            self.dx.scale is not None
            and self.dy.scale is not None
            and _is_symmetric(self.a)
            and _is_symmetric(self.b)
        )


def tridiagonal_matvec(coefficients: Tridiagonal, x: torch.Tensor) -> torch.Tensor:
    """y = T x для пакета трёхдиагональных матриц (вдоль последней оси)."""
    lower, diagonal, upper = coefficients
    y = diagonal * x
    y[:, 1:] += lower[:, 1:] * x[:, :-1]
    y[:, :-1] += upper[:, :-1] * x[:, 1:]
    return y


class AdiPreconditioner:
    """
    Предобусловливатель P ≈ M⁻¹ из тех же одномерных прогонок.

    При скалярных DX = dx·I, DY = dy·I обозначим X = A/dx (вдоль строк),
    Y = B/dy (вдоль столбцов). Тогда
        M = X⁻¹ + Y⁻¹ = X⁻¹ (X + Y) Y⁻¹,   M⁻¹ = Y (X + Y)⁻¹ X.
    (X + Y)⁻¹ — двумерная задача; приближаем её m шагами ADI
    (Писмен — Рэкфорд) с геометрическими параметрами τ_1..τ_m:
        (X + τ) w½ = f - (Y - τ) w,     (Y + τ) w = f - (X - τ) w½.
    Каждый шаг — одна прогонка по строкам и одна по столбцам.
    Итого P z = Y · ADI_m(X z): 2m прогонок на одно применение.

    Для постоянных коэффициентов X и Y коммутируют, и P симметричен;
    в общем случае P — просто хороший предобусловливатель для GMRES.
    """

    def __init__(
        self,
        problem: CoordinatorProblem,
        steps: int = 6,
        bounds: tuple[float, float] | None = None,
    ) -> None:
        if problem.dx.scale is None or problem.dy.scale is None:
            raise ValueError("ADI preconditioner needs scalar DX and DY")

        self.problem = problem
        self.dx, self.dy = problem.dx.scale, problem.dy.scale
        self.x_coef = tuple(t / self.dx for t in problem.a)
        self.y_coef = tuple(t / self.dy for t in problem.b)

        alpha, beta = bounds if bounds is not None else self._spectral_bounds()
        # Геометрические параметры между границами спектра.
        self.taus = [
            beta * (alpha / beta) ** ((2 * j - 1) / (2 * steps)) for j in range(1, steps + 1)
        ]
        self.x_factors = [self._shifted_factor(self.x_coef, tau) for tau in self.taus]
        self.y_factors = [self._shifted_factor(self.y_coef, tau) for tau in self.taus]

    def _shifted_factor(self, coef: Tridiagonal, tau: float) -> TridiagonalFactor:
        lower, diagonal, upper = coef
        return TridiagonalFactor(lower, diagonal + tau, upper, method=self.problem.solver)

    MAX_SPECTRUM_ROWS = 16

    def _spectral_bounds(self) -> tuple[float, float]:
        """
        Границы спектра X и Y по собственным числам одномерных матриц.

        Постоянные коэффициенты -> одна матрица. Переменные -> не более
        MAX_SPECTRUM_ROWS равномерно выбранных строк: τ — лишь параметры
        ускорения, GMRES остаётся точным при любых τ.
        Если lower·upper > 0 (диффузия + противопоточная адвекция), матрица
        диагональным подобием сводится к симметричной с внедиагональю
        √(lower·upper) -> eigvalsh, спектр вещественный.
        """
        lows, highs = [], []
        for coef in (self.x_coef, self.y_coef):
            lower, diagonal, upper = (t.cpu().double() for t in coef)
            # Одинаковые строки (постоянные коэффициенты) -> одна матрица.
            if all(torch.equal(t, t[:1].expand_as(t)) for t in (lower, diagonal, upper)):
                lower, diagonal, upper = lower[:1], diagonal[:1], upper[:1]
            elif diagonal.shape[0] > self.MAX_SPECTRUM_ROWS:
                pick = torch.linspace(0, diagonal.shape[0] - 1, self.MAX_SPECTRUM_ROWS).round().long()
                lower, diagonal, upper = lower[pick], diagonal[pick], upper[pick]
            idx = torch.arange(diagonal.shape[1])
            dense = torch.diag_embed(diagonal)
            products = lower[:, 1:] * upper[:, :-1]
            if bool(torch.all(products > 0)):
                off = torch.sqrt(products)
                dense[:, idx[1:], idx[:-1]] = off
                dense[:, idx[:-1], idx[1:]] = off
                eig = torch.linalg.eigvalsh(dense)
            else:
                dense[:, idx[1:], idx[:-1]] = lower[:, 1:]
                dense[:, idx[:-1], idx[1:]] = upper[:, :-1]
                eig = torch.linalg.eigvals(dense).real
            lows.append(eig.min().item())
            highs.append(eig.max().item())
        alpha = min(lows)
        if alpha <= 0.0:
            raise ValueError("X and Y must be positive definite")
        return alpha, max(highs)

    def _x(self, w):
        return tridiagonal_matvec(self.x_coef, w)

    def _y(self, w):
        return tridiagonal_matvec(self.y_coef, w.T).T

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        p = self.problem
        f = self._x(z)
        w = torch.zeros_like(f)
        for tau, fx, fy in zip(self.taus, self.x_factors, self.y_factors):
            rhs = f - self._y(w) + tau * w
            p.tridiagonal_solves += 1
            w_half = fx.solve(rhs)
            rhs = f - self._x(w_half) + tau * w_half
            p.tridiagonal_solves += 1
            w = fy.solve(rhs.T).T
        return self._y(w)


def functional(r: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(r * r)


def grad_adjoint(
    problem: CoordinatorProblem,
    q: torch.Tensor,
    r: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ∇J = -(DXᵀp + DYᵀs),  Aᵀp = r,  Bᵀs = r.

    Если r уже известна (из рекурсии), прямая прогонка не нужна.
    """
    if r is None:
        r = problem.residual(q)
    return -problem.apply_m_t(r)


# ---------------------------------------------------------------------------
# Общая инфраструктура методов
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    method: str
    q: torch.Tensor
    r: torch.Tensor
    iterations: int
    converged: bool
    relative_measure: float  # ‖g‖/‖g₀‖ (SD, CGLS) или ‖r‖/‖r₀‖ (CR, GMRES)
    history: list[float]  # J по итерациям
    tridiagonal_solves: int = 0
    seconds: float = 0.0


class _Monitor:
    """Монотонность J, лог и подсчёт стоимости для одного запуска."""

    def __init__(self, method: str, problem: CoordinatorProblem, r0: torch.Tensor,
                 log_every: int) -> None:
        self.method = method
        self.problem = problem
        self.log_every = log_every
        self.solves_at_start = problem.tridiagonal_solves
        self.start = time.perf_counter()
        self.history = [functional(r0).item()]

    def step(self, j_new: float, iteration: int) -> None:
        j_old = self.history[-1]
        if j_new > j_old * (1.0 + 1e-12):
            raise RuntimeError(
                f"{self.method}: J increased at iteration {iteration}: "
                f"{j_old:.16e} -> {j_new:.16e}"
            )
        self.history.append(j_new)

    def log(self, iteration: int, rel: float) -> None:
        if self.log_every and iteration % self.log_every == 0:
            print(
                f"{self.method} {iteration:05d} | J: {self.history[-1]:.8e} "
                f"| rel: {rel:.3e}"
            )

    def result(self, q, r, iterations, converged, rel) -> SolveResult:
        return SolveResult(
            method=self.method,
            q=q,
            r=r,
            iterations=iterations,
            converged=converged,
            relative_measure=rel,
            history=self.history,
            tridiagonal_solves=self.problem.tridiagonal_solves - self.solves_at_start,
            seconds=time.perf_counter() - self.start,
        )


def _dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y)


# ---------------------------------------------------------------------------
# Наискорейший спуск и CGLS (минимизация J через градиент)
# ---------------------------------------------------------------------------


def steepest_descent(
    problem: CoordinatorProblem,
    q0: torch.Tensor,
    rtol: float = 1e-10,
    max_iter: int = 10_000,
    log_every: int = 0,
) -> SolveResult:
    """Точный шаг α = ‖g‖² / ‖Mg‖²; остановка ‖g‖/‖g₀‖ < rtol."""
    q = q0.clone()
    r = problem.residual(q)
    mon = _Monitor("SD", problem, r, log_every)
    g = grad_adjoint(problem, q, r)

    g0_norm = torch.linalg.vector_norm(g).item()
    if g0_norm == 0.0:
        return mon.result(q, r, 0, True, 0.0)

    rel = 1.0
    for iteration in range(max_iter):
        rel = torch.linalg.vector_norm(g).item() / g0_norm
        mon.log(iteration, rel)
        if rel < rtol:
            return mon.result(q, r, iteration, True, rel)

        mg = problem.apply_m(g)
        alpha = _dot(g, g) / _dot(mg, mg)

        q = q - alpha * g
        r = r + alpha * mg  # r = c - Mq, q ← q - αg
        mon.step(functional(r).item(), iteration + 1)

        g = grad_adjoint(problem, q, r)

    rel = torch.linalg.vector_norm(g).item() / g0_norm
    return mon.result(q, r, max_iter, rel < rtol, rel)


def cgls(
    problem: CoordinatorProblem,
    q0: torch.Tensor,
    rtol: float = 1e-10,
    max_iter: int = 10_000,
    log_every: int = 0,
) -> SolveResult:
    """CGLS для min ‖Mq - c‖; s = Mᵀr = -∇J; остановка ‖s‖/‖s₀‖ < rtol."""
    q = q0.clone()
    r = problem.residual(q)
    mon = _Monitor("CGLS", problem, r, log_every)
    s = problem.apply_m_t(r)
    direction = s.clone()

    gamma = _dot(s, s)
    g0_norm = math.sqrt(gamma.item())
    if g0_norm == 0.0:
        return mon.result(q, r, 0, True, 0.0)

    rel = 1.0
    for iteration in range(max_iter):
        rel = math.sqrt(gamma.item()) / g0_norm
        mon.log(iteration, rel)
        if rel < rtol:
            return mon.result(q, r, iteration, True, rel)

        t = problem.apply_m(direction)
        alpha = gamma / _dot(t, t)

        q = q + alpha * direction
        r = r - alpha * t
        mon.step(functional(r).item(), iteration + 1)

        s = problem.apply_m_t(r)
        gamma_new = _dot(s, s)
        direction = s + (gamma_new / gamma) * direction
        gamma = gamma_new

    rel = math.sqrt(gamma.item()) / g0_norm
    return mon.result(q, r, max_iter, rel < rtol, rel)


# ---------------------------------------------------------------------------
# Krylov на квадратной системе M q = c (r = c - Mq = u - v)
# ---------------------------------------------------------------------------


def conjugate_residual(
    problem: CoordinatorProblem,
    q0: torch.Tensor,
    rtol: float = 1e-10,
    max_iter: int = 10_000,
    log_every: int = 0,
) -> SolveResult:
    """
    Метод сопряжённых невязок для M = Mᵀ > 0.

    Минимизирует ‖r‖ на подпространстве Крылова → J монотонно убывает.
    Одно умножение на M (2 прогонки) за итерацию, сходимость ∝ √κ(M).
    """
    if not problem.m_is_symmetric():
        raise ValueError("conjugate_residual needs symmetric M; use gmres")

    q = q0.clone()
    r = problem.residual(q)
    mon = _Monitor("CR", problem, r, log_every)

    r0_norm = torch.linalg.vector_norm(r).item()
    if r0_norm == 0.0:
        return mon.result(q, r, 0, True, 0.0)

    mr = problem.apply_m(r)
    p, mp = r.clone(), mr.clone()
    r_mr = _dot(r, mr)

    rel = 1.0
    for iteration in range(max_iter):
        rel = torch.linalg.vector_norm(r).item() / r0_norm
        mon.log(iteration, rel)
        if rel < rtol:
            return mon.result(q, r, iteration, True, rel)

        if r_mr.item() <= 0.0:
            raise RuntimeError("CR: rᵀMr <= 0, M is not positive definite")

        alpha = r_mr / _dot(mp, mp)
        q = q + alpha * p
        r = r - alpha * mp
        mon.step(functional(r).item(), iteration + 1)

        mr = problem.apply_m(r)
        r_mr_new = _dot(r, mr)
        beta = r_mr_new / r_mr
        p = r + beta * p
        mp = mr + beta * mp
        r_mr = r_mr_new

    rel = torch.linalg.vector_norm(r).item() / r0_norm
    return mon.result(q, r, max_iter, rel < rtol, rel)


def gmres(
    problem: CoordinatorProblem,
    q0: torch.Tensor,
    rtol: float = 1e-10,
    max_iter: int = 10_000,
    restart: int = 100,
    log_every: int = 0,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> SolveResult:
    """
    GMRES(restart) для M q = c при любой невырожденной M.

    Минимизирует ‖r‖ на подпространстве Крылова → J монотонно убывает
    (и внутри цикла, и между перезапусками). Одно умножение на M за итерацию.
    После каждого цикла невязка пересчитывается прямой задачей.

    preconditioner P — правое предобусловливание: решаем M P y = c, q = P y.
    Минимизируется по-прежнему настоящая невязка ‖c - Mq‖ = ‖u - v‖,
    поэтому монотонность J сохраняется.
    """
    q = q0.clone()
    r = problem.residual(q)
    name = "GMRES" if preconditioner is None else "GMRES+P"
    mon = _Monitor(name, problem, r, log_every)

    r0_norm = torch.linalg.vector_norm(r).item()
    if r0_norm == 0.0:
        return mon.result(q, r, 0, True, 0.0)

    shape = q.shape
    total = 0
    rel = 1.0

    while total < max_iter:
        beta = torch.linalg.vector_norm(r).item()
        rel = beta / r0_norm
        if rel < rtol:
            return mon.result(q, r, total, True, rel)

        m = min(restart, max_iter - total)
        basis = torch.zeros((m + 1, r.numel()), device=r.device, dtype=r.dtype)
        basis[0] = r.reshape(-1) / beta
        search = basis if preconditioner is None else torch.zeros_like(basis)
        h = torch.zeros((m + 1, m), dtype=torch.float64)
        cs = torch.zeros(m, dtype=torch.float64)
        sn = torch.zeros(m, dtype=torch.float64)
        rhs = torch.zeros(m + 1, dtype=torch.float64)
        rhs[0] = beta

        k_used = 0
        for k in range(m):
            if preconditioner is not None:
                search[k] = preconditioner(basis[k].reshape(shape)).reshape(-1)
            w = problem.apply_m(search[k].reshape(shape)).reshape(-1)

            # Классический Грам — Шмидт дважды (CGS2): устойчиво и одной matmul.
            for _ in range(2):
                coeffs = basis[: k + 1] @ w
                w = w - basis[: k + 1].T @ coeffs
                h[: k + 1, k] += coeffs.cpu().to(torch.float64)
            h_next = torch.linalg.vector_norm(w).item()
            h[k + 1, k] = h_next

            # Предыдущие вращения Гивенса к новому столбцу.
            for i in range(k):
                hi, hi1 = h[i, k].item(), h[i + 1, k].item()
                h[i, k] = cs[i] * hi + sn[i] * hi1
                h[i + 1, k] = -sn[i] * hi + cs[i] * hi1

            a_, b_ = h[k, k].item(), h[k + 1, k].item()
            denom = math.hypot(a_, b_)
            cs[k], sn[k] = a_ / denom, b_ / denom
            h[k, k] = denom
            h[k + 1, k] = 0.0
            rhs[k + 1] = -sn[k] * rhs[k]
            rhs[k] = cs[k] * rhs[k]

            total += 1
            k_used = k + 1
            residual_estimate = abs(rhs[k + 1].item())
            mon.step(0.5 * residual_estimate**2, total)
            rel = residual_estimate / r0_norm
            mon.log(total, rel)

            if rel < rtol or h_next == 0.0:
                break
            basis[k + 1] = w / h_next

        y = torch.linalg.solve_triangular(
            h[:k_used, :k_used], rhs[:k_used].unsqueeze(1), upper=True
        ).squeeze(1)
        update = search[:k_used].T @ y.to(device=r.device, dtype=r.dtype)
        q = q + update.reshape(shape)

        # Истинная невязка (защита от накопления ошибок в оценке).
        r = problem.residual(q)
        mon.history[-1] = functional(r).item()

    rel = torch.linalg.vector_norm(r).item() / r0_norm
    return mon.result(q, r, total, rel < rtol, rel)


def coordinate(
    problem: CoordinatorProblem,
    q0: torch.Tensor,
    rtol: float = 1e-10,
    max_iter: int = 10_000,
    log_every: int = 0,
) -> SolveResult:
    """
    Самый эффективный доступный метод:
      DX, DY скалярные  -> GMRES + ADI-предобусловливание;
      M = Mᵀ            -> CR;
      иначе             -> GMRES.
    """
    if problem.dx.scale is not None and problem.dy.scale is not None:
        try:
            pre = AdiPreconditioner(problem)
        except ValueError:
            pre = None
        if pre is not None:
            return gmres(problem, q0, rtol, max_iter, log_every=log_every, preconditioner=pre)
    if problem.m_is_symmetric():
        return conjugate_residual(problem, q0, rtol, max_iter, log_every)
    return gmres(problem, q0, rtol, max_iter, log_every=log_every)


# ---------------------------------------------------------------------------
# Задача Лапласа на прямоугольнике
# ---------------------------------------------------------------------------


def build_tridiagonal(
    batch_size: int,
    system_size: int,
    device: str,
    dtype: torch.dtype,
) -> Tridiagonal:
    lower = -torch.ones((batch_size, system_size), device=device, dtype=dtype)
    diagonal = 2 * torch.ones((batch_size, system_size), device=device, dtype=dtype)
    upper = -torch.ones((batch_size, system_size), device=device, dtype=dtype)

    lower[:, 0] = 0
    upper[:, -1] = 0

    return lower, diagonal, upper


def sine_top(x: torch.Tensor, lx: float) -> torch.Tensor:
    return torch.sin(torch.pi * x / lx)


def build_laplace_problem(
    nx: int,
    ny: int,
    device: str,
    dtype: torch.dtype,
    lx: float = 1.0,
    ly: float = 1.0,
    top: Callable[[torch.Tensor, float], torch.Tensor] = sine_top,
) -> CoordinatorProblem:
    """
    Прямоугольник [0, lx] x [0, ly], nx x ny узлов вместе с границей.

    -u_xx = -q,  -v_yy = q,  u(x, ly) = top(x), остальные границы 0.

    После умножения на шаги:  A u = -hx² q,  B v = hy² q + FY,
    т.е. DX = hx²·I, DY = hy²·I.
    q: [ny-2, nx-2]; строка q[i, :] — узлы с одним y, столбец — с одним x.
    A: [ny-2, nx-2] (системы вдоль x), B: [nx-2, ny-2] (системы вдоль y).
    """
    nx_in, ny_in = nx - 2, ny - 2
    hx = lx / (nx - 1)
    hy = ly / (ny - 1)

    x = torch.linspace(0.0, lx, nx, device=device, dtype=dtype)

    fx = torch.zeros((ny_in, nx_in), device=device, dtype=dtype)
    fy = torch.zeros((ny_in, nx_in), device=device, dtype=dtype)
    # Последний внутренний узел по y связан с верхней границей y = ly.
    fy[-1, :] = top(x[1:-1], lx)

    return CoordinatorProblem(
        a=build_tridiagonal(ny_in, nx_in, device, dtype),
        b=build_tridiagonal(nx_in, ny_in, device, dtype),
        dx=LinearMap(scale=hx * hx),
        dy=LinearMap(scale=hy * hy),
        fx=fx,
        fy=fy,
    )


def assemble_solution(
    problem: CoordinatorProblem,
    q: torch.Tensor,
    lx: float = 1.0,
    top: Callable[[torch.Tensor, float], torch.Tensor] = sine_top,
) -> torch.Tensor:
    """Полное решение [ny, nx]: внутри ½(u + v), сверху top(x), остальное 0."""
    u, v = problem.states(q)
    ny_in, nx_in = u.shape
    solution = torch.zeros((ny_in + 2, nx_in + 2), device=u.device, dtype=u.dtype)
    solution[1:-1, 1:-1] = 0.5 * (u + v)
    x = torch.linspace(0.0, lx, nx_in + 2, device=u.device, dtype=u.dtype)
    solution[-1, 1:-1] = top(x[1:-1], lx)
    return solution


def main() -> None:
    device = get_device()
    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64
    rtol = 1e-5 if dtype == torch.float32 else 1e-10

    # Прямоугольник [0, 2] x [0, 1], hx = hy = 1/60.
    lx, ly = 2.0, 1.0
    nx, ny = 121, 61

    problem = build_laplace_problem(nx, ny, device, dtype, lx=lx, ly=ly)
    q0 = torch.zeros_like(problem.fx)

    synchronize(device)
    start = time.perf_counter()

    result = coordinate(problem, q0, rtol=rtol, max_iter=10_000, log_every=5)

    synchronize(device)
    elapsed = time.perf_counter() - start

    u, v = problem.states(result.q)

    torch.save(result.q.cpu(), "coordinator_q.pt")
    torch.save(u.cpu(), "coordinator_horizontal.pt")
    torch.save(v.cpu(), "coordinator_vertical.pt")

    print(f"\n=== Coordinator method for Laplace equation ({result.method}) ===")
    print(f"Device: {device}")
    print(f"Domain: [0, {lx}] x [0, {ly}], grid {nx} x {ny} (x by y)")
    print(f"Iterations: {result.iterations}")
    print(f"Tridiagonal solves: {result.tridiagonal_solves}")
    print(f"Converged: {result.converged} (||r||/||r0|| = {result.relative_measure:.3e})")
    print(f"Final J(q): {functional(u - v).item():.8e}")
    print(f"Max |u - v|: {torch.max(torch.abs(u - v)).item():.8e}")
    print(f"Time: {elapsed:.4f} sec")
    print("Saved: coordinator_q.pt")
    print("Saved: coordinator_horizontal.pt")
    print("Saved: coordinator_vertical.pt")


if __name__ == "__main__":
    main()
