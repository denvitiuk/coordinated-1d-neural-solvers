"""
Ниша «подзадачи — нейросети»: насколько координация усиливает ошибки одномерных решателей?

Нейросеть-заменитель одномерной задачи никогда не точна. Моделируем это так:
одномерный решатель отдаёт X⁻¹r с относительной ошибкой ε,

    Ñ(r) = X⁻¹r ⊙ (1 + ε·z),

где z — фиксированное поле (решатель «ошибается одинаково», как обученная сеть).
Два вида ошибки:
    white  — z случайно в каждом узле (шум);
    smooth — z = гладкая функция (систематическое смещение, типично для сетей).
Затем координация решается до конца (GMRES + ADI) с неточными решателями,
и сравнивается с точным решением разностной схемы.

Ключевой вопрос: коэффициент усиления = (ошибка решения) / ε.
    ~1–10, не растёт с сеткой      -> ниша жива;
    растёт как 1/h² (сотни-тысячи) -> координация с нейросетями бессмысленна.

    python surrogate_error_robustness.py
"""

from __future__ import annotations

import math

import torch

from coordinator_laplace_descent import AdiPreconditioner, assemble_solution, build_laplace_problem, gmres
from laplace_gpu_benchmark import exact_discrete, top_values


class NoisySolver:
    """Точная прогонка, выход которой искажён на относительную величину ε·z."""

    def __init__(self, exact, z: torch.Tensor, eps: float) -> None:
        self.exact, self.z, self.eps = exact, z, eps

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        return self.exact.solve(rhs) * (1.0 + self.eps * self.z)


def error_field(kind: str, shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "white":
        return torch.randn(shape, generator=g, dtype=torch.float64)
    rows, cols = shape
    s = torch.linspace(0, 1, cols, dtype=torch.float64)
    t = torch.linspace(0, 1, rows, dtype=torch.float64)
    a, b, c = torch.randn(3, generator=g, dtype=torch.float64)
    # Гладкое смещение порядка 1: сеть «тянет» решение в одну сторону на участках.
    return (a * torch.sin(math.pi * s)[None, :] + b * torch.cos(2 * math.pi * t)[:, None]
            + c * torch.sin(3 * math.pi * s)[None, :] * torch.sin(math.pi * t)[:, None])


def run(n: int, eps: float, kind: str) -> tuple[float, int]:
    pr = build_laplace_problem(n, n, "cpu", torch.float64, top=top_values)
    pre = AdiPreconditioner(pr)  # предобусловливатель точный: он ускоряет, но не меняет ответ
    shape = pr.fx.shape
    pr._a = NoisySolver(pr._a, error_field(kind, shape, 1), eps)
    pr._b = NoisySolver(pr._b, error_field(kind, (shape[1], shape[0]), 2), eps)
    res = gmres(pr, torch.zeros_like(pr.fx), rtol=1e-12, max_iter=300, preconditioner=pre)
    phi = assemble_solution(pr, res.q, top=top_values)[1:-1, 1:-1]
    exact = exact_discrete(n)[1:-1, 1:-1]
    rel = (torch.linalg.vector_norm(phi - exact) / torch.linalg.vector_norm(exact)).item()
    return rel, res.iterations


def main() -> None:
    print("Ошибка решения (отн., норма L2) и усиление = ошибка / ε")
    print(f"{'вид':>6} | {'сетка':>9} | " + " | ".join(f"ε={e:.0e}: ошибка  усил." for e in (1e-1, 1e-2, 1e-3)))
    for kind in ("white", "smooth"):
        for n in (50, 100, 200, 400):
            cells = []
            for eps in (1e-1, 1e-2, 1e-3):
                rel, it = run(n, eps, kind)
                cells.append(f"{rel:9.2e} {rel / eps:6.2f} ({it:2d} it)")
            print(f"{kind:>6} | {n:>4}x{n:<4} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
