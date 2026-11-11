"""
Опыт 2 ниши «подзадачи — нейросети»: координация, в которой ВСЕ одномерные решения
делает обученная нейросеть, а не прогонка.

Задача: −∇·(k ∇φ) = 0 в [0,1]², φ = 1 сверху, 0 на остальных сторонах,
k(x, y) > 0 — гладкое случайное поле. Расщепление по направлениям:
    X = −∂x(k ∂x) по строкам,  Y = −∂y(k ∂y) по столбцам,
    X u = g_x − q,   Y v = g_y + q,   u = v  ⇔  (X + Y) φ = g_x + g_y.
(Всё в единицах «матрица без 1/h²»: A u = r.)

Одна сеть N(c, r) ≈ A(c)⁻¹ r для одномерной задачи −(c u′)′ = r с нулевыми
краевыми условиями; c — коэффициент в полуузлах линии. Та же сеть решает
и строки, и столбцы. Однородность по r и по c встроена точно:
    N(c, r) = ‖r‖ / mean(c) · net(log(c / mean c), r / ‖r‖).

Координация: невязка R(q) = N(kx, g_x − q) − N(ky, g_y + q)ᵀ нелинейна (сеть),
решаем R(q) = 0 методом Ньютона–Крылова: Якобиан — через torch.func.jvp,
внутренний решатель — GMRES без предобусловливания (сеть — «чёрный ящик»).

Вопрос: ошибка итогового φ против ошибки самой сети ε на её реальных входах.

    python nn_coordination.py
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from batched_tridiagonal import TridiagonalFactor
from coordinator_laplace_descent import gmres

N_GRID = 50          # узлов с границей
M = N_GRID - 2       # внутренних узлов на линии
H = 1.0 / (N_GRID - 1)
CHECKPOINTS = (300, 1000, 3000, 10000, 20000)
BATCH = 512
BOUNDARY_SAMPLES = True  # False — исходный запуск (без граничных пиков в обучении)
torch.manual_seed(0)


# --- одномерная задача ----------------------------------------------------

def tridiag_from_halves(c: torch.Tensor):
    """−(c u′)′ без 1/h²: c [B, M+1] в полуузлах -> (lower, diag, upper) [B, M]."""
    lower = -c[:, :-1].clone()
    upper = -c[:, 1:].clone()
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    return lower, c[:, :-1] + c[:, 1:], upper


def exact_solve(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return TridiagonalFactor(*tridiag_from_halves(c.double()), method="thomas").solve(r.double())


def smooth_1d(batch: int, n: int, gen: torch.Generator, modes: int = 6) -> torch.Tensor:
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    k = torch.arange(1, modes + 1, dtype=torch.float64)
    a = torch.randn(batch, modes, generator=gen, dtype=torch.float64) / k
    phase = torch.rand(batch, modes, generator=gen, dtype=torch.float64) * 2 * math.pi
    return torch.einsum("bm,bmn->bn", a, torch.sin(math.pi * k[None, :, None] * x[None, None, :] + phase[..., None]))


def sample_batch(batch: int, gen: torch.Generator):
    """Случайные коэффициенты и правые части разной гладкости."""
    c = torch.exp(0.8 * smooth_1d(batch, M + 1, gen, modes=4))
    x = torch.linspace(0, 1, M + 2, dtype=torch.float64)[1:-1]
    kk = torch.arange(1, M + 1, dtype=torch.float64)
    p = torch.rand(batch, 1, generator=gen, dtype=torch.float64) * 2.5
    amp = torch.randn(batch, M, generator=gen, dtype=torch.float64) * kk[None, :] ** (-p)
    r = amp @ torch.sin(math.pi * kk[:, None] * x[None, :])
    # Иногда — сосредоточенный источник у края (как от граничного условия).
    spike = torch.rand(batch, generator=gen) < 0.2
    pos = torch.randint(0, M, (batch,), generator=gen)
    r[spike, pos[spike]] += 5 * torch.randn(int(spike.sum()), generator=gen, dtype=torch.float64)
    if BOUNDARY_SAMPLES:
        # Граничные данные: пик в первом или последнем узле поверх малой гладкой части
        # (так выглядят столбцы у верхней границы φ = 1 и строки у боковых границ).
        edge = torch.rand(batch, generator=gen) < 0.4
        ne = int(edge.sum())
        small = 10 ** (-3 * torch.rand(ne, 1, generator=gen, dtype=torch.float64))
        r[edge] = r[edge] / torch.linalg.vector_norm(r[edge], dim=1, keepdim=True) * small
        last = torch.rand(ne, generator=gen) < 0.5
        col = torch.where(last, M - 1, 0)
        r[edge, col] += torch.randn(ne, generator=gen, dtype=torch.float64).sign() * (0.5 + torch.rand(ne, generator=gen, dtype=torch.float64))
    return c, r


class LineSolver(nn.Module):
    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * M + 1, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, M),
        )

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        cm = c.mean(dim=1, keepdim=True)
        s = torch.linalg.vector_norm(r, dim=1, keepdim=True).clamp_min(1e-30)
        z = self.net(torch.cat([torch.log(c / cm), r / s], dim=1))
        # Масштаб выхода: решение −u″ = r имеет порядок r·n²/π², сеть учит O(1).
        return z * s / cm * (M + 1) ** 2 / 100.0


def rel_err(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a - b, dim=-1) / torch.linalg.vector_norm(b, dim=-1)


# --- двумерная задача -----------------------------------------------------

def coefficient_field(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    ks = torch.randn(6, generator=g, dtype=torch.float64)
    def kfun(x, y):
        return torch.exp(0.6 * (ks[0] * torch.sin(math.pi * x) * torch.sin(math.pi * y)
                                + ks[1] * torch.cos(2 * math.pi * x) + ks[2] * torch.cos(2 * math.pi * y)
                                + 0.5 * ks[3] * torch.sin(3 * math.pi * x + ks[4]) * torch.cos(math.pi * y + ks[5])))
    nodes = torch.arange(1, M + 1, dtype=torch.float64) * H
    halves = (torch.arange(0, M + 1, dtype=torch.float64) + 0.5) * H
    kx = kfun(halves[None, :], nodes[:, None])          # [строка i (y), полуузел по x]
    ky = kfun(nodes[:, None], halves[None, :])          # [столбец j (x), полуузел по y]
    gx = torch.zeros(M, M, dtype=torch.float64)
    gy = torch.zeros(M, M, dtype=torch.float64)
    gy[-1, :] = ky[:, -1] * 1.0                        # φ = 1 на верхней границе
    return kx, ky, gx, gy


def reference(kx, ky, gx, gy) -> torch.Tensor:
    """Прямое решение (X + Y) φ = g_x + g_y плотной матрицей (2 304 неизвестных)."""
    n = M * M
    a = torch.zeros(n, n, dtype=torch.float64)
    idx = lambda i, j: i * M + j
    for i in range(M):
        lo, di, up = (t[0] for t in tridiag_from_halves(kx[i:i + 1]))
        for j in range(M):
            a[idx(i, j), idx(i, j)] += di[j]
            if j > 0: a[idx(i, j), idx(i, j - 1)] += lo[j]
            if j < M - 1: a[idx(i, j), idx(i, j + 1)] += up[j]
    for j in range(M):
        lo, di, up = (t[0] for t in tridiag_from_halves(ky[j:j + 1]))
        for i in range(M):
            a[idx(i, j), idx(i, j)] += di[i]
            if i > 0: a[idx(i, j), idx(i - 1, j)] += lo[i]
            if i < M - 1: a[idx(i, j), idx(i + 1, j)] += up[i]
    return torch.linalg.solve(a, (gx + gy).reshape(-1)).reshape(M, M)


class NewtonStep:
    """Линейная задача шага Ньютона для gmres(): (−J) δ = R(q)."""

    def __init__(self, fun, q, r0):
        # GMRES ведём во float64 (ортогонализация), сеть считает во float32.
        self.fun, self.q, self.r0 = fun, q.float(), r0.double()
        self.tridiagonal_solves = 0

    def apply_m(self, d):
        return -torch.func.jvp(self.fun, (self.q,), (d.float(),))[1].detach().double()

    def residual(self, d):
        return self.r0 - self.apply_m(d)


def coordinate(solver, kx, ky, gx, gy, tol=1e-6, newton_max=8, preconditioner=None):
    # Веса сети на время решения заморожены: иначе каждый вектор Крылова тянет граф обучения.
    flags = [p.requires_grad for p in solver.parameters()]
    for p in solver.parameters():
        p.requires_grad_(False)
    try:
        return _coordinate(solver, kx, ky, gx, gy, tol, newton_max, preconditioner)
    finally:
        for p, f in zip(solver.parameters(), flags):
            p.requires_grad_(f)


def _coordinate(solver, kx, ky, gx, gy, tol, newton_max, preconditioner):
    kx32, ky32, gx32, gy32 = (t.float() for t in (kx, ky, gx, gy))

    def states(q):
        u = solver(kx32, gx32 - q)
        v = solver(ky32, (gy32 + q).T).T
        return u, v

    def resid(q):
        u, v = states(q)
        return u - v

    q = torch.zeros(M, M)
    r = resid(q)
    r_first = torch.linalg.vector_norm(r).item()
    total_inner = 0
    for _ in range(newton_max):
        if torch.linalg.vector_norm(r).item() < tol * r_first:
            break
        step = NewtonStep(resid, q, r)
        # Без перезапусков: шум float32 в Якобиане ломает монотонность между циклами.
        res = gmres(step, torch.zeros(M, M, dtype=torch.float64), rtol=1e-4, max_iter=1500, restart=1500,
                    preconditioner=preconditioner)
        total_inner += res.iterations
        q = q + res.q.float()
        r = resid(q)
    u, v = states(q)
    return (0.5 * (u + v)).double(), torch.linalg.vector_norm(r).item() / r_first, total_inner, q, u, v


def main() -> None:
    kx, ky, gx, gy = coefficient_field()
    phi_ref = reference(kx, ky, gx, gy)
    print(f"k от {min(kx.min(), ky.min()).item():.2f} до {max(kx.max(), ky.max()).item():.2f}; "
          f"эталон: плотное решение {M}² = {M * M} неизвестных")

    solver = LineSolver()
    opt = torch.optim.Adam(solver.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CHECKPOINTS[-1], eta_min=1e-5)
    gen = torch.Generator().manual_seed(1)
    test_c, test_r = sample_batch(4096, torch.Generator().manual_seed(99))
    test_u = exact_solve(test_c, test_r)

    print(f"{'шагов':>6} | {'ε сети (тест)':>13} | {'ε сети (свои входы)':>19} | "
          f"{'ошибка φ':>9} | {'φ / ε':>6} | {'Ньютон: невязка, итер. GMRES':>28} | {'время':>6}")
    t0 = time.perf_counter()
    for step in range(1, CHECKPOINTS[-1] + 1):
        c, r = sample_batch(BATCH, gen)
        u = exact_solve(c, r)
        loss = rel_err(solver(c.float(), r.float()), u.float()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step in CHECKPOINTS:
            solver.eval()
            with torch.no_grad():
                eps_test = rel_err(solver(test_c.float(), test_r.float()).double(), test_u).mean().item()
            phi, rel_res, inner, q, u_nn, v_nn = coordinate(solver, kx, ky, gx, gy)
            # Ошибка сети на тех самых линиях, которые она решала в найденном решении.
            q64 = q.double()
            u_true = exact_solve(kx, gx - q64)
            v_true = exact_solve(ky, (gy + q64).T).T
            eps_own = 0.5 * ((torch.linalg.vector_norm(u_nn.double() - u_true) / torch.linalg.vector_norm(u_true))
                             + (torch.linalg.vector_norm(v_nn.double() - v_true) / torch.linalg.vector_norm(v_true))).item()
            err = (torch.linalg.vector_norm(phi - phi_ref) / torch.linalg.vector_norm(phi_ref)).item()
            print(f"{step:>6} | {eps_test:13.2e} | {eps_own:19.2e} | {err:9.2e} | {err / eps_own:6.2f} | "
                  f"{rel_res:12.1e}, {inner:6d}           | {time.perf_counter() - t0:5.0f} с", flush=True)
            solver.train()

    torch.save(solver.state_dict(), f"results/nn_line_solver{'_bnd' if BOUNDARY_SAMPLES else ''}.pt")


if __name__ == "__main__":
    main()
