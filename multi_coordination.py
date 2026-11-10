"""
Общий метод координации: K независимых подсистем с общей целью.

    A_k u_k = D_k q + F_k,        k = 1..K   (каждая решается своим решателем)
    z_k = E_k u_k                            (что подсистема «показывает» наружу:
                                              всё решение или только общую зону)
    J(q) = ½ Σ_{i<j} ‖z_i - z_j‖²            (при K = 2: ½‖z_1 - z_2‖², как в дипломе)

Подсистемы — «чёрные ящики»: координатору нужно только уметь решить
A_k x = b и A_kᵀ x = b. Сами подсистемы могут считаться на разных машинах
(«Варшава» и «Киев»): между ними и координатором ходят только векторы
q, z_k и сопряжённые невязки — сами матрицы никуда не передаются.

Линейная алгебра. Σ_{i<j}‖z_i - z_j‖² = K Σ_k ‖z_k - z̄‖², поэтому
    r = √K · (z_k - z̄)_k   (K блоков),   J = ½‖r‖²,   r = c - M q,
    M g    = -√K · center( E_k A_k⁻¹ D_k g )_k,
    Mᵀ y   = -√K · Σ_k D_kᵀ A_k⁻ᵀ E_kᵀ center(y)_k,
    ∇J     = -Mᵀ r       (по одной сопряжённой задаче на подсистему).
M прямоугольная (K блоков на один q), поэтому основной метод — CGLS.
Все методы из coordinator_laplace_descent (SD, CGLS) работают без изменений.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from coordinator_laplace_descent import SolveResult, cgls

Tensor = torch.Tensor


class Operator:
    """Линейный оператор с известным транспонированным."""

    def __init__(self, apply: Callable[[Tensor], Tensor], apply_t: Callable[[Tensor], Tensor]):
        self.apply = apply
        self.apply_t = apply_t

    @staticmethod
    def identity() -> "Operator":
        return Operator(lambda x: x, lambda x: x)

    @staticmethod
    def matrix(mat: Tensor, in_shape: Sequence[int], out_shape: Sequence[int]) -> "Operator":
        """Плотная матрица [out, in] над сплющенными тензорами заданных форм."""
        in_shape, out_shape = tuple(in_shape), tuple(out_shape)
        return Operator(
            lambda x: (mat @ x.reshape(-1)).reshape(out_shape),
            lambda y: (mat.T @ y.reshape(-1)).reshape(in_shape),
        )


@dataclass
class Subsystem:
    """
    A u = D q + F как чёрный ящик.

    solve(b)   решает A x = b,
    solve_t(b) решает Aᵀ x = b,
    d          оператор q -> правая часть,
    f          постоянная правая часть,
    observe    E: решение -> общее пространство (по умолчанию всё решение).
    """

    solve: Callable[[Tensor], Tensor]
    solve_t: Callable[[Tensor], Tensor]
    d: Operator
    f: Tensor
    observe: Operator = field(default_factory=Operator.identity)
    name: str = ""

    @staticmethod
    def dense(a: Tensor, d: Operator, f: Tensor, observe: Operator | None = None,
              name: str = "") -> "Subsystem":
        """Подсистема с плотной матрицей: LU один раз, дальше только подстановки."""
        lu, piv = torch.linalg.lu_factor(a)
        shape = f.shape

        def solve(b: Tensor) -> Tensor:
            return torch.linalg.lu_solve(lu, piv, b.reshape(-1, 1)).reshape(shape)

        def solve_t(b: Tensor) -> Tensor:
            return torch.linalg.lu_solve(lu, piv, b.reshape(-1, 1), adjoint=True).reshape(shape)

        return Subsystem(solve, solve_t, d, f, observe or Operator.identity(), name)


class MultiCoordinationProblem:
    """Задача координации K подсистем; интерфейс как у CoordinatorProblem."""

    def __init__(self, subsystems: Sequence[Subsystem]) -> None:
        if len(subsystems) < 2:
            raise ValueError("need at least two subsystems")
        self.subsystems = list(subsystems)
        self.k = len(subsystems)
        self.scale = math.sqrt(self.k)
        self.tridiagonal_solves = 0  # здесь: число решений подсистем (любого типа)

    def _center(self, blocks: list[Tensor]) -> Tensor:
        z = torch.stack(blocks)
        return z - z.mean(dim=0, keepdim=True)

    def states(self, q: Tensor) -> list[Tensor]:
        self.tridiagonal_solves += self.k
        return [s.solve(s.d.apply(q) + s.f) for s in self.subsystems]

    def residual(self, q: Tensor) -> Tensor:
        z = [s.observe.apply(u) for s, u in zip(self.subsystems, self.states(q))]
        return self.scale * self._center(z)

    def apply_m(self, g: Tensor) -> Tensor:
        self.tridiagonal_solves += self.k
        z = [s.observe.apply(s.solve(s.d.apply(g))) for s in self.subsystems]
        return -self.scale * self._center(z)

    def apply_m_t(self, y: Tensor) -> Tensor:
        self.tridiagonal_solves += self.k
        yc = y - y.mean(dim=0, keepdim=True)
        out = None
        for s, block in zip(self.subsystems, yc):
            term = s.d.apply_t(s.solve_t(s.observe.apply_t(block)))
            out = term if out is None else out + term
        return -self.scale * out

    def m_is_symmetric(self) -> bool:
        return False

    def consensus(self, q: Tensor) -> Tensor:
        """Согласованный ответ: среднее показаний подсистем."""
        z = [s.observe.apply(u) for s, u in zip(self.subsystems, self.states(q))]
        return torch.stack(z).mean(dim=0)


def coordinate_many(
    problem: MultiCoordinationProblem,
    q0: Tensor,
    rtol: float = 1e-10,
    max_iter: int = 10_000,
    log_every: int = 0,
) -> SolveResult:
    """CGLS: сходимость ∝ κ(M), одна прямая + одна сопряжённая задача на подсистему за итерацию."""
    return cgls(problem, q0, rtol=rtol, max_iter=max_iter, log_every=log_every)


# ---------------------------------------------------------------------------
# Пример «Варшава — Киев»: две перекрывающиеся области с общей зоной
# ---------------------------------------------------------------------------


def laplace_dense(nx_in: int, ny_in: int, hx: float, hy: float, dtype=torch.float64) -> Tensor:
    """5-точечный -Δ (Дирихле) на сетке nx_in x ny_in, порядок [y, x] row-major."""
    def lap1(n, h):
        m = 2.0 * torch.eye(n, dtype=dtype)
        m -= torch.diag(torch.ones(n - 1, dtype=dtype), 1)
        m -= torch.diag(torch.ones(n - 1, dtype=dtype), -1)
        return m / h**2

    return torch.kron(torch.eye(ny_in, dtype=dtype), lap1(nx_in, hx)) + torch.kron(
        lap1(ny_in, hy), torch.eye(nx_in, dtype=dtype)
    )


@dataclass
class OverlapDemo:
    """
    Прямоугольник [0, lx] x [0, ly], -ΔT = 0, T = top(x) сверху, 0 на остальных сторонах.
    Сетка nx x ny. «Варшава» — столбцы 0..right1, «Киев» — столбцы left2..nx-1,
    left2 < right1, общая зона — столбцы left2..right1.

    Неизвестные q = (значения T на искусственной правой границе Варшавы,
                     значения T на искусственной левой границе Киева).
    Каждая область решает свою задачу Дирихле сама; цель — совпадение в общей зоне.
    """

    lx: float = 2.0
    ly: float = 1.0
    nx: int = 41
    ny: int = 21
    right1: int = 24
    left2: int = 16
    top: Callable[[Tensor, float], Tensor] = lambda x, lx: torch.sin(math.pi * x / lx)

    def build(self, dtype=torch.float64) -> MultiCoordinationProblem:
        hx, hy = self.lx / (self.nx - 1), self.ly / (self.ny - 1)
        ny_in = self.ny - 2
        x = torch.linspace(0.0, self.lx, self.nx, dtype=dtype)
        top_vals = self.top(x, self.lx)
        n_q = 2 * ny_in
        overlap = list(range(self.left2 + 1, self.right1))  # внутренние столбцы общей зоны

        def region(first: int, last: int, q_slice: slice, artificial_side: str, name: str):
            cols = list(range(first + 1, last))  # внутренние столбцы области
            nx_in = len(cols)
            a = laplace_dense(nx_in, ny_in, hx, hy, dtype)

            f = torch.zeros((ny_in, nx_in), dtype=dtype)
            f[-1, :] += top_vals[cols] / hy**2  # верхняя граница (физическая)

            # q входит как данные Дирихле на искусственной границе.
            d = torch.zeros((ny_in * nx_in, n_q), dtype=dtype)
            j = nx_in - 1 if artificial_side == "right" else 0
            for i in range(ny_in):
                d[i * nx_in + j, q_slice.start + i] = 1.0 / hx**2

            # Наблюдение: значения в общей зоне.
            sel = torch.zeros((ny_in * len(overlap), ny_in * nx_in), dtype=dtype)
            for i in range(ny_in):
                for k, c in enumerate(overlap):
                    sel[i * len(overlap) + k, i * nx_in + cols.index(c)] = 1.0

            return Subsystem.dense(
                a,
                Operator.matrix(d, (n_q,), (ny_in, nx_in)),
                f,
                Operator.matrix(sel, (ny_in, nx_in), (ny_in, len(overlap))),
                name,
            ), cols

        warsaw, self.cols1 = region(0, self.right1, slice(0, ny_in), "right", "Warsaw")
        kyiv, self.cols2 = region(self.left2, self.nx - 1, slice(ny_in, n_q), "left", "Kyiv")
        self.n_q = n_q
        return MultiCoordinationProblem([warsaw, kyiv])

    def global_solution(self, dtype=torch.float64) -> Tensor:
        """Прямое решение всей области (эталон), внутренние узлы [ny-2, nx-2]."""
        hx, hy = self.lx / (self.nx - 1), self.ly / (self.ny - 1)
        nx_in, ny_in = self.nx - 2, self.ny - 2
        x = torch.linspace(0.0, self.lx, self.nx, dtype=dtype)
        f = torch.zeros((ny_in, nx_in), dtype=dtype)
        f[-1, :] = self.top(x, self.lx)[1:-1] / hy**2
        a = laplace_dense(nx_in, ny_in, hx, hy, dtype)
        return torch.linalg.solve(a, f.reshape(-1)).reshape(ny_in, nx_in)
