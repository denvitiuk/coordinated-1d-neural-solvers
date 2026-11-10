"""
Задача из диплома: стационарная теплопроводность в пластине [0, ALX] x [0, ALY]
решением координацией двух семейств одномерных задач.

    -T_xx - T_yy = 0,
    ∂T/∂n = 0 при x = 0 и x = ALX (теплоизоляция, T_0 = T_1 как в main.py),
    T(x, ALY) = TU·cos(πx/ALX),   T(x, 0) = TD·sin(πx/ALX).

Почему нужен сдвиг s
--------------------
По x у каждой строки условия Неймана с обеих сторон, поэтому матрица L_x
вырождена (константа в ядре): система (7) диплома  -u_xx = -q  не имеет
единственного решения. Сдвигаем обе задачи на одну и ту же величину s:

    (L_x + s) u = -q,          (L_y - s) v = q + g.

При u = v сдвиги сокращаются и сумма даёт ровно 5-точечную схему (6).
При 0 < s < λ_min(L_y) обе матрицы симметричны и положительно определены,
M = (L_x + s)⁻¹ + (L_y - s)⁻¹ симметрична и > 0 -> метод CR.
Наибольшее собственное число M минимально при s = λ_min(L_y) / 2.

Умножив на шаги:  A u = -hx² q,  B v = hy² q + FY,  т.е. DX = hx²·I, DY = hy²·I.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from batched_tridiagonal import get_device, synchronize
from coordinator_laplace_descent import (
    CoordinatorProblem,
    LinearMap,
    SolveResult,
    coordinate,
)


@dataclass
class Plate:
    alx: float = 5.0
    aly: float = 12.0
    mx: int = 51  # узлов по x вместе с границей
    ny: int = 121  # узлов по y вместе с границей
    tu: float = 60.0
    td: float = 20.0

    @property
    def hx(self) -> float:
        return self.alx / (self.mx - 1)

    @property
    def hy(self) -> float:
        return self.aly / (self.ny - 1)

    def top(self, x: torch.Tensor) -> torch.Tensor:
        return self.tu * torch.cos(math.pi * x / self.alx)

    def bottom(self, x: torch.Tensor) -> torch.Tensor:
        return self.td * torch.sin(math.pi * x / self.alx)

    def lambda_min_y(self) -> float:
        """Наименьшее собственное число разностного -d²/dy² (Дирихле)."""
        return (2.0 - 2.0 * math.cos(math.pi / (self.ny - 1))) / self.hy**2

    def default_shift(self) -> float:
        return 0.5 * self.lambda_min_y()


def build_plate_problem(
    plate: Plate,
    device: str,
    dtype: torch.dtype,
    shift: float | None = None,
) -> CoordinatorProblem:
    """
    q: [ny-2, nx-2]; строка 0 — y = hy (у нижней границы), последняя — у верхней.
    A: [ny-2, nx-2] — системы вдоль x (Нейман), B: [nx-2, ny-2] — вдоль y (Дирихле).
    """
    s = plate.default_shift() if shift is None else shift
    if not 0.0 < s < plate.lambda_min_y():
        raise ValueError(f"shift must be in (0, {plate.lambda_min_y():.6g}), got {s}")

    nx_in, ny_in = plate.mx - 2, plate.ny - 2
    hx2, hy2 = plate.hx**2, plate.hy**2

    def tridiagonal(batch, size, diag_value, end_value):
        lower = -torch.ones((batch, size), device=device, dtype=dtype)
        upper = -torch.ones((batch, size), device=device, dtype=dtype)
        diag = torch.full((batch, size), diag_value, device=device, dtype=dtype)
        diag[:, 0] = end_value
        diag[:, -1] = end_value
        lower[:, 0] = 0
        upper[:, -1] = 0
        return lower, diag, upper

    # По x: T_0 = T_1 -> на краях диагональ 1 вместо 2; плюс сдвиг s·hx².
    a = tridiagonal(ny_in, nx_in, 2.0 + s * hx2, 1.0 + s * hx2)
    # По y: Дирихле сверху и снизу; минус сдвиг s·hy².
    b = tridiagonal(nx_in, ny_in, 2.0 - s * hy2, 2.0 - s * hy2)

    x = torch.linspace(0.0, plate.alx, plate.mx, device=device, dtype=dtype)[1:-1]
    fy = torch.zeros((ny_in, nx_in), device=device, dtype=dtype)
    fy[0, :] += plate.bottom(x)
    fy[-1, :] += plate.top(x)

    return CoordinatorProblem(
        a=a,
        b=b,
        dx=LinearMap(scale=hx2),
        dy=LinearMap(scale=hy2),
        fx=torch.zeros((ny_in, nx_in), device=device, dtype=dtype),
        fy=fy,
    )


def assemble_plate(plate: Plate, problem: CoordinatorProblem, q: torch.Tensor) -> torch.Tensor:
    """Полное поле T [ny, mx] (строка 0 — низ), как ZZ в main.py диплома."""
    u, v = problem.states(q)
    t = torch.zeros((plate.ny, plate.mx), device=u.device, dtype=u.dtype)
    t[1:-1, 1:-1] = 0.5 * (u + v)
    t[1:-1, 0] = t[1:-1, 1]  # T_0 = T_1
    t[1:-1, -1] = t[1:-1, -2]
    x = torch.linspace(0.0, plate.alx, plate.mx, device=u.device, dtype=u.dtype)
    t[-1, :] = plate.top(x)
    t[0, :] = plate.bottom(x)
    return t


def exact_plate(plate: Plate, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Точное решение той же разностной схемы без координатора:
    разложение по собственным векторам L_x (Нейман, eigh) + 1D задачи по y.
    """
    nx_in, ny_in = plate.mx - 2, plate.ny - 2
    hx2, hy2 = plate.hx**2, plate.hy**2

    lx = 2.0 * torch.eye(nx_in, dtype=dtype)
    lx -= torch.diag(torch.ones(nx_in - 1, dtype=dtype), 1)
    lx -= torch.diag(torch.ones(nx_in - 1, dtype=dtype), -1)
    lx[0, 0] = lx[-1, -1] = 1.0
    eigvals, eigvecs = torch.linalg.eigh(lx / hx2)  # [мода], [узел, мода]

    x = torch.linspace(0.0, plate.alx, plate.mx, dtype=dtype)[1:-1]
    top_m = eigvecs.T @ plate.top(x)
    bottom_m = eigvecs.T @ plate.bottom(x)

    ly = 2.0 * torch.eye(ny_in, dtype=dtype)
    ly -= torch.diag(torch.ones(ny_in - 1, dtype=dtype), 1)
    ly -= torch.diag(torch.ones(ny_in - 1, dtype=dtype), -1)
    mats = ly.unsqueeze(0) / hy2 + eigvals[:, None, None] * torch.eye(ny_in, dtype=dtype)
    rhs = torch.zeros((nx_in, ny_in, 1), dtype=dtype)
    rhs[:, 0, 0] = bottom_m / hy2
    rhs[:, -1, 0] = top_m / hy2
    w = torch.linalg.solve(mats, rhs).squeeze(2)  # [мода, y]

    t = torch.zeros((plate.ny, plate.mx), dtype=dtype)
    t[1:-1, 1:-1] = w.T @ eigvecs.T
    t[1:-1, 0] = t[1:-1, 1]
    t[1:-1, -1] = t[1:-1, -2]
    xf = torch.linspace(0.0, plate.alx, plate.mx, dtype=dtype)
    t[-1, :] = plate.top(xf)
    t[0, :] = plate.bottom(xf)
    return t


def solve_plate(
    plate: Plate,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    shift: float | None = None,
    rtol: float = 1e-10,
    log_every: int = 0,
) -> tuple[torch.Tensor, SolveResult, CoordinatorProblem]:
    problem = build_plate_problem(plate, device, dtype, shift)
    result = coordinate(problem, torch.zeros_like(problem.fx), rtol=rtol, log_every=log_every)
    return assemble_plate(plate, problem, result.q), result, problem


def main() -> None:
    device = get_device()
    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64
    rtol = 1e-6 if dtype == torch.float32 else 1e-10

    plate = Plate()  # 5 x 12, сетка 51 x 121, как в тексте диплома

    synchronize(device)
    start = time.perf_counter()
    temperature, result, problem = solve_plate(plate, device, dtype, rtol=rtol, log_every=20)
    synchronize(device)
    elapsed = time.perf_counter() - start

    u, v = problem.states(result.q)
    exact = exact_plate(plate)

    print(f"\n=== Plate {plate.alx:g} x {plate.aly:g}, grid {plate.mx} x {plate.ny} ===")
    print(f"Device: {device}, method: {result.method}, shift s = {plate.default_shift():.6g}")
    print(f"Unknowns: {(plate.mx - 2) * (plate.ny - 2)} = "
          f"{plate.ny - 2} systems of {plate.mx - 2} + {plate.mx - 2} systems of {plate.ny - 2}")
    print(f"Iterations: {result.iterations}")
    print(f"Tridiagonal solves: {result.tridiagonal_solves}")
    print(f"Converged: {result.converged}")
    print(f"Max |u - v|: {torch.max(torch.abs(u - v)).item():.3e}")
    print(f"Max |T - T_direct|: {torch.max(torch.abs(temperature.cpu().double() - exact)).item():.3e}")
    print(f"Time: {elapsed:.4f} sec")

    # Сохранение: поле T [ny, mx] (строка 0 — низ) и тепловая карта.
    from pathlib import Path

    import numpy as np

    from rectangle_report import heatmap, write_png

    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    name = f"plate_{plate.mx}x{plate.ny}"
    field = temperature.cpu().double().numpy()
    np.save(results / f"{name}.npy", field)
    write_png(results / f"{name}.png", heatmap(field[::-1]), 4)
    print(f"Saved: results/{name}.npy, results/{name}.png")


if __name__ == "__main__":
    main()
