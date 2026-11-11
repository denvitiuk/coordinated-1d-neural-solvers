"""
Опыт 4 ниши «подзадачи — нейросети»: одна одномерная сеть для ЛЮБОЙ сетки.

Одномерный решатель с обычным интерфейсом «чёрного ящика»:
    −(c u′)′ = f на (0, 1),  u(0) = a,  u(1) = b,
вход — c в полуузлах, f в узлах (физические единицы), a, b; выход — u в узлах.
Внутри: u = u_hom + w. u_hom — решение однородной задачи с краевыми a, b
(в 1D точно: поток постоянен, u_hom = a + (b − a)·Σ1/c / Σ1/c), w — нейросеть
с нулевыми краевыми условиями. Сеть — FNO по синусам (спектральные слои в
базисе sin kπx): её веса не зависят от числа узлов, поэтому её можно учить
на коротких линиях и применять на длинных.

Координация на сетке N × N:  −(k u_x)_x = −q  по строкам,  −(k v_y)_y = q  по
столбцам, граничные значения — со сторон квадрата; u = v  ⇔  −∇·(k∇φ) = 0.
R(q) = u(q) − v(q) = 0 — Ньютон–Крылов (jvp) с грубым ADI-предобусловливателем.
Эталон — та же координация с точной прогонкой (линейная, GMRES до 1e-11).

    python nn_coordination_grids.py          # обучит сеть (если нет весов) и проверит
"""

from __future__ import annotations

import math
import os
import time

import torch
from torch import nn

from adi_direct_vs_coordination import DirectAdi
from batched_tridiagonal import TridiagonalFactor
from coordinator_laplace_descent import gmres

NET = "hybrid"            # "hybrid" (v2) или "fno" (v1, не прошёл в 2D)
WEIGHTS = "results/nn_line_fno.pt" if NET == "fno" else "results/nn_line_hybrid.pt"
P_MIN = 0.5 if NET == "fno" else 0.0   # 0 — «плоский» спектр f, нужен для высоких частот
TRAIN_N = (24, 96)        # длины линий (внутренние узлы) при обучении
STEPS = 8000 if NET == "fno" else 40000
BATCH = 256
MODES = 32
WIDTH = 64
LAYERS = 4
torch.manual_seed(0)


# --- точный одномерный решатель (для обучения и эталона) -------------------

def tridiag(c: torch.Tensor):
    lower = -c[:, :-1].clone()
    upper = -c[:, 1:].clone()
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    return lower, c[:, :-1] + c[:, 1:], upper


def exact_line(c, f, a, b):
    """c [B, n+1], f [B, n], a, b [B] -> u [B, n]; разностная схема второго порядка."""
    c, f, a, b = (t.double() for t in (c, f, a, b))
    n = f.shape[1]
    h = 1.0 / (n + 1)
    r = h * h * f
    r[:, 0] += c[:, 0] * a
    r[:, -1] += c[:, -1] * b
    return TridiagonalFactor(*tridiag(c), method="thomas").solve(r)


def homogeneous(c, a, b):
    """Точное дискретное решение −(c u′)′ = 0, u(0) = a, u(1) = b, в узлах."""
    inv = 1.0 / c
    s = torch.cumsum(inv, dim=1)[:, :-1] / inv.sum(dim=1, keepdim=True)
    return a[:, None] + (b - a)[:, None] * s


# --- сеть ---------------------------------------------------------------

def sine_basis(n: int, modes: int, device, dtype):
    x = torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)
    k = torch.arange(1, min(modes, n) + 1, device=device, dtype=dtype)
    s = torch.sin(math.pi * k[:, None] * x[None, :])         # [K, n]
    return s * (2.0 / (n + 1)), s.T, x                         # прямое (≈ 2∫·sin), обратное


class SpectralLayer(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.r = nn.Parameter(torch.randn(modes, width, width) / width)
        self.w = nn.Linear(width, width)

    def forward(self, h, fwd, inv):
        k = fwd.shape[0]
        coef = torch.einsum("kn,bnw->bkw", fwd, h)
        coef = torch.einsum("bkw,kwv->bkv", coef, self.r[:k])
        return nn.functional.gelu(torch.einsum("nk,bkv->bnv", inv, coef) + self.w(h))


class LineFNO(nn.Module):
    def __init__(self, width=WIDTH, modes=MODES, layers=LAYERS) -> None:
        super().__init__()
        self.modes = modes
        self.lift = nn.Linear(4, width)
        self.layers = nn.ModuleList(SpectralLayer(width, modes) for _ in range(layers))
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def zero_bc(self, c, f):
        n = f.shape[1]
        fwd, inv, x = sine_basis(n, self.modes, f.device, f.dtype)
        cm = c.mean(dim=1, keepdim=True)
        logc = torch.log(c / cm)
        logc_nodes = 0.5 * (logc[:, :-1] + logc[:, 1:])
        s = f.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-30)
        feats = torch.stack([logc_nodes, f / s, x.expand_as(f), 0.5 * (logc[:, 1:] - logc[:, :-1]) * (n + 1) / 10],
                            dim=-1)
        h = self.lift(feats)
        for layer in self.layers:
            h = layer(h, fwd, inv)
        return self.proj(h)[..., 0] * s / cm * 0.1

    def forward(self, c, f, a, b):
        return homogeneous(c, a, b) + self.zero_bc(c, f)


class LineHybrid(nn.Module):
    """
    Сеть v2: w = P₀(c) f + S_Kᵀ · G_θ(c) · S_K f.

    P₀ = h² D^{-1/2} T⁻¹ D^{-1/2} (D = c в узлах) — дешёвая аналитическая основа,
    точная на высоких частотах (там ответ определяет c рядом с точкой).
    G_θ(c) — матрица K × K для первых K синус-мод, её выдаёт MLP по коэффициентам
    log c; она исправляет низкие частоты, где важна вся линия.
    По f отображение ЛИНЕЙНО (как и настоящий решатель), по c — нелинейно.
    Синус-коэффициенты нормированы как интегралы, поэтому длина линии любая.
    """

    def __init__(self, modes: int = MODES, c_modes: int = 16, width: int = 512) -> None:
        super().__init__()
        self.modes, self.c_modes = modes, c_modes
        self.mlp = nn.Sequential(nn.Linear(c_modes + 1, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, modes * modes))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def prior(c, f):
        n = f.shape[1]
        h = 1.0 / (n + 1)
        x = torch.arange(1, n + 1, device=f.device, dtype=f.dtype) / (n + 1)
        k = torch.arange(1, n + 1, device=f.device, dtype=f.dtype)
        s = torch.sin(math.pi * k[:, None] * x[None, :])
        lam = 4 * torch.sin(k * math.pi / (2 * (n + 1))) ** 2
        d = (0.5 * (c[:, :-1] + c[:, 1:])).sqrt()
        y = (f / d) @ s.T / lam * (2.0 / (n + 1))
        return h * h * (y @ s) / d

    def zero_bc(self, c, f):
        n = f.shape[1]
        fwd, inv, x = sine_basis(n, self.modes, f.device, f.dtype)
        kk = fwd.shape[0]
        cm = c.mean(dim=1, keepdim=True)
        logc = torch.log(0.5 * (c[:, :-1] + c[:, 1:]) / cm)
        j = torch.arange(0, self.c_modes, device=f.device, dtype=f.dtype)
        feats = logc @ torch.cos(math.pi * j[None, :] * x[:, None]) * (1.0 / (n + 1))
        g = self.mlp(torch.cat([feats, torch.log(cm)], dim=1)).view(-1, self.modes, self.modes)[:, :kk, :kk]
        coef = f @ fwd.T                                         # [B, K]
        corr = torch.einsum("bij,bj->bi", g, coef) / (math.pi ** 2) / cm
        return self.prior(c, f) + corr @ inv.T

    def forward(self, c, f, a, b):
        return homogeneous(c, a, b) + self.zero_bc(c, f)


# --- обучающие задачи (непрерывные функции -> любая сетка) ------------------

def sample(batch: int, n: int, gen: torch.Generator):
    xh = (torch.arange(0, n + 1, dtype=torch.float64) + 0.5) / (n + 1)
    x = torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)
    m = torch.arange(1, 5, dtype=torch.float64)
    am = torch.randn(batch, 4, generator=gen, dtype=torch.float64) / m
    ph = torch.rand(batch, 4, generator=gen, dtype=torch.float64) * 2 * math.pi
    c = torch.exp(1.0 * torch.einsum("bm,bmn->bn", am, torch.sin(math.pi * m[None, :, None] * xh + ph[..., None])))

    k = torch.arange(1, 65, dtype=torch.float64)
    p = P_MIN + (2.5 - P_MIN) * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    amp = torch.randn(batch, 64, generator=gen, dtype=torch.float64) * k ** (-p)
    f = amp @ torch.sin(math.pi * k[:, None] * x[None, :])
    # Источник, сосредоточенный у края (у угла квадрата координатор q так и выглядит).
    edge = torch.rand(batch, generator=gen) < 0.3
    d = 10 ** (-2 + torch.rand(batch, 1, generator=gen, dtype=torch.float64))
    side = torch.where(torch.rand(batch, 1, generator=gen) < 0.5, x[None, :], 1 - x[None, :])
    bump = torch.exp(-side / d) / d * torch.randn(batch, 1, generator=gen, dtype=torch.float64)
    f = torch.where(edge[:, None], f / f.pow(2).mean(1, keepdim=True).sqrt() * 0.1 + bump, f)
    a = torch.randn(batch, generator=gen, dtype=torch.float64) * (torch.rand(batch, generator=gen) < 0.5)
    b = torch.randn(batch, generator=gen, dtype=torch.float64) * (torch.rand(batch, generator=gen) < 0.5)
    return c, f, a, b


def rel(a, b):
    return torch.linalg.vector_norm(a - b, dim=-1) / torch.linalg.vector_norm(b, dim=-1).clamp_min(1e-30)


def make_net():
    return LineFNO() if NET == "fno" else LineHybrid()


def apply_a(c, w):
    """A w (без 1/h²) для −(c w′)′ с нулевыми краевыми."""
    lower, diag, upper = tridiag(c)
    y = diag * w
    y[:, 1:] += lower[:, 1:] * w[:, :-1]
    y[:, :-1] += upper[:, :-1] * w[:, 1:]
    return y


def train():
    net = make_net()
    lr = 2e-3 if NET == "fno" else 5e-4
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.05)
    gen = torch.Generator().manual_seed(1)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        n = int(torch.randint(TRAIN_N[0], TRAIN_N[1] + 1, (1,), generator=gen))
        c, f, a, b = sample(BATCH, n, gen)
        w_true = (exact_line(c, f, a, b) - homogeneous(c, a, b)).float()
        w = net.zero_bc(c.float(), f.float())
        loss = rel(w, w_true).mean()
        if NET == "hybrid":
            # Ошибка в невязке: A(w − w*) — так высокие частоты весят столько же, сколько низкие.
            cf = c.float()
            loss = loss + rel(apply_a(cf, w), apply_a(cf, w_true)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"   шаг {step:5d}: ошибка {loss.item():.2%}, {time.perf_counter() - t0:4.0f} с", flush=True)
    torch.save(net.state_dict(), WEIGHTS)
    return net


# --- двумерная задача ------------------------------------------------------

def problem2d(n_grid: int, seed: int, bc: str, contrast: float = 0.6):
    g = torch.Generator().manual_seed(seed)
    ks = torch.randn(6, generator=g, dtype=torch.float64)

    def k(x, y):
        return torch.exp(contrast * (ks[0] * torch.sin(math.pi * x) * torch.sin(math.pi * y)
                                     + ks[1] * torch.cos(2 * math.pi * x) + ks[2] * torch.cos(2 * math.pi * y)
                                     + 0.5 * ks[3] * torch.sin(3 * math.pi * x + ks[4]) * torch.cos(math.pi * y + ks[5])))

    m = n_grid - 2
    h = 1.0 / (n_grid - 1)
    nodes = torch.arange(1, m + 1, dtype=torch.float64) * h
    halves = (torch.arange(0, m + 1, dtype=torch.float64) + 0.5) * h
    kx = k(halves[None, :], nodes[:, None])     # [строка (y), полуузел по x]
    ky = k(nodes[:, None], halves[None, :])     # [столбец (x), полуузел по y]
    zero = torch.zeros(m, dtype=torch.float64)
    left, right, bottom, top = zero, zero.clone(), zero.clone(), zero.clone()
    if bc == "top=1":
        top = torch.ones(m, dtype=torch.float64)
    elif bc == "top=sin":
        top = torch.sin(math.pi * nodes)
    elif bc == "right=1, низ горб":
        right = torch.ones(m, dtype=torch.float64)
        bottom = 0.5 * (1 - torch.cos(2 * math.pi * nodes))
    return dict(kx=kx, ky=ky, left=left, right=right, bottom=bottom, top=top, h=h, m=m)


def residual_fn(solve, p, dtype):
    kx, ky = p["kx"].to(dtype), p["ky"].to(dtype)
    l, r, bt, tp = (p[s].to(dtype) for s in ("left", "right", "bottom", "top"))

    def states(q):
        u = solve(kx, -q, l, r)
        v = solve(ky, q.T, bt, tp).T
        return u, v

    return states


class Precond:
    """(−J)⁻¹ ≈ k̄/h² · T_y (T_x + T_y)⁻¹ T_x: ADI для постоянного k̄."""

    def __init__(self, p) -> None:
        self.scale = 0.5 * (p["kx"].mean() + p["ky"].mean()).item() / p["h"] ** 2
        self.b = torch.zeros(p["m"], p["m"], dtype=torch.float64)
        self.tridiagonal_solves = 0
        self.adi = DirectAdi(self, p["m"], "thomas")

    def __call__(self, z):
        return self.scale * self.adi._t(self.adi(self.adi._t(z)).T).T


class Linearized:
    """Задача для gmres(): residual(d) = r0 − (−J) d, apply_m(d) = −J d."""

    def __init__(self, fun, q, r0, jvp: bool) -> None:
        self.fun, self.q, self.r0, self.jvp = fun, q, r0, jvp
        self.tridiagonal_solves = 0

    def apply_m(self, d):
        if self.jvp:
            return -torch.func.jvp(self.fun, (self.q,), (d.to(self.q.dtype),))[1].detach().double()
        return (self.r0_base - self.fun(self.q + d)).double()  # линейный случай

    def residual(self, d):
        return self.r0 - self.apply_m(d)


def coordinate(solve, p, dtype, tol, newton_max=6, rtol_lin=1e-4):
    states = residual_fn(solve, p, dtype)

    def resid(q):
        u, v = states(q)
        return u - v

    q = torch.zeros(p["m"], p["m"], dtype=dtype)
    r = resid(q)
    r_first = torch.linalg.vector_norm(r).item()
    inner = 0
    pre = Precond(p)
    for _ in range(newton_max):
        if torch.linalg.vector_norm(r).item() < tol * r_first:
            break
        lin = Linearized(resid, q, r.double(), jvp=dtype == torch.float32)
        lin.r0_base = r
        res = gmres(lin, torch.zeros(p["m"], p["m"], dtype=torch.float64), rtol=rtol_lin,
                    max_iter=800, restart=800, preconditioner=pre)
        inner += res.iterations
        q = q + res.q.to(dtype)
        r = resid(q)
    u, v = states(q)
    return (0.5 * (u + v)).double(), inner, torch.linalg.vector_norm(r).item() / r_first


def main() -> None:
    if os.environ.get("LINE") == "refined":      # LINE=refined: сеть v3 с уточнением (nn_line_refined.py)
        import nn_line_refined
        net = nn_line_refined.load()
        print(f"Сеть v3 из {nn_line_refined.WEIGHTS}, уточнений R = {net.refine}")
    elif os.path.exists(WEIGHTS):
        net = make_net()
        net.load_state_dict(torch.load(WEIGHTS))
        print(f"Сеть загружена из {WEIGHTS}")
    else:
        print(f"Обучение: линии длиной {TRAIN_N[0]}–{TRAIN_N[1]} узлов, {STEPS} шагов по {BATCH}")
        net = train()
    net.eval()
    # v2 линейна по f: координация — одна линейная система, считаем её во float64.
    dt = torch.float64 if NET == "hybrid" else torch.float32
    net = net.to(dt)
    for prm in net.parameters():
        prm.requires_grad_(False)
    print(f"Параметров сети: {sum(p.numel() for p in net.parameters()):,}")

    print("\n1D: ошибка сети на линиях разной длины (обучение — 24…96):")
    gen = torch.Generator().manual_seed(77)
    for n in (32, 64, 128, 256, 512):
        c, f, a, b = sample(1024, n, gen)
        with torch.no_grad():
            u = net(c.to(dt), f.to(dt), a.to(dt), b.to(dt)).double()
        w_true = exact_line(c, f, a, b) - homogeneous(c, a, b)
        print(f"   n = {n:4d}: ошибка w {rel(u - homogeneous(c, a, b), w_true).mean():.2%}, "
              f"ошибка u {rel(u, exact_line(c, f, a, b)).mean():.2%}", flush=True)

    print("\n1D: ошибка по частотам, f = sin kπx (n = 48 и 198):")
    for n in (48, 198):
        c, _, _, _ = sample(64, n, gen)
        x = torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)
        z = torch.zeros(64, dtype=torch.float64)
        row = []
        for k in (1, 4, 16, n // 2, n - 1):
            f = torch.sin(math.pi * k * x).expand(64, n).clone()
            with torch.no_grad():
                w = net.zero_bc(c.to(dt), f.to(dt)).double()
            row.append(f"k={k}: {rel(w, exact_line(c, f, z, z)).mean():.1%}")
        print(f"   n = {n}: " + " | ".join(row), flush=True)

    print("\n2D: координация, где ВСЕ строки и столбцы решает сеть (без дообучения):")
    print(f"   {'сетка':>9} | {'seed':>4} | {'граничные условия':18} | {'ошибка φ':>8} | "
          f"{'итер.':>5} | {'сеть':>6} | {'прогонка':>8}")
    for n_grid in (50, 100, 200):
        errs = []
        for seed in (21, 22, 23):
            for bc in ("top=1", "top=sin", "right=1, низ горб"):
                p = problem2d(n_grid, seed, bc)
                t0 = time.perf_counter()
                phi_ref, it_ref, _ = coordinate(exact_line, p, torch.float64, tol=1e-11, newton_max=1,
                                                rtol_lin=1e-12)
                t_ref = time.perf_counter() - t0
                t0 = time.perf_counter()
                if NET == "hybrid":
                    phi, it, res = coordinate(net, p, torch.float64, tol=1e-10, newton_max=1, rtol_lin=1e-10)
                else:
                    phi, it, res = coordinate(net, p, torch.float32, tol=1e-5)
                t_nn = time.perf_counter() - t0
                err = (torch.linalg.vector_norm(phi - phi_ref) / torch.linalg.vector_norm(phi_ref)).item()
                errs.append(err)
                print(f"   {n_grid:>4}x{n_grid:<4} | {seed:>4} | {bc:18} | {err:8.1e} | {it:5d} | "
                      f"{t_nn:5.1f}с | {t_ref:6.2f}с", flush=True)
        print(f"   {'':>9}   средняя {sum(errs) / len(errs):.2e}, худшая {max(errs):.2e}\n", flush=True)


if __name__ == "__main__":
    main()
