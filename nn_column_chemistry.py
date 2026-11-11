"""
Опыт 6 ниши «подзадачи — нейросети»: нелинейная «колонка» атмосферы — нейросеть,
горизонтальный перенос — обычная схема, связь — координация.

Вертикальный разрез атмосферы [0,1]² (x — горизонталь, z — высота), установившаяся
концентрация загрязнителя φ ≥ 0:
    −(k φ_x)_x − (K φ_z)_z + σ φ|φ| = s,   φ = 0 на границе,
s — источник (труба у земли + площадной источник), σ(x) — скорость химической
реакции второго порядка (зависит от освещённости), K(x, z) — вертикальное
турбулентное перемешивание, k(x, z) — горизонтальное.

Расщепление (как в моделях погоды и климата: горизонталь отдельно, колонки отдельно):
    строки:   −(k u_x)_x                 = −q        (линейно, прогонка)
    колонки:  −(K v_z)_z + σ v|v|         = s + q     (НЕЛИНЕЙНО: нейросеть)
u = v  ⇔  исходное уравнение. Нелинейная система R(q) = u − v = 0 — Ньютон–Крылов,
Якобиан сети — через torch.func.jvp, предобусловливатель — грубый ADI (k̄, K̄).

Колонку без сети решают методом Ньютона (несколько прогонок на колонку). Сеть
обучена ТОЛЬКО на отдельных колонках; двумерных задач она не видела.

    python nn_column_chemistry.py        # обучит колонку (если нет весов) и проверит
"""

from __future__ import annotations

import math
import os
import time

import torch
from torch import nn

import nn_coordination_grids as ng
from batched_tridiagonal import TridiagonalFactor
from coordinator_laplace_descent import gmres

VERSION = "mu"              # "mu" (v2) или "direct" (v1: 15–26 % ошибки колонки)
WEIGHTS = "results/nn_column_chem.pt" if VERSION == "mu" else "results/nn_column_chem_v1.pt"
STEPS = 30000
BATCH = 256
MODES = 32
torch.manual_seed(0)


# --- точная колонка: Ньютон по прогонке ------------------------------------

def exact_column(kc, sigma, g, return_factor=False):
    """−(K v′)′ + σ v|v| = g, v = 0 на концах. kc [B, n+1], sigma [B], g [B, n]."""
    kc, sigma, g = kc.double(), sigma.double(), g.double()
    n = g.shape[1]
    h2 = 1.0 / (n + 1) ** 2
    lower, diag, upper = ng.tridiag(kc)
    v = torch.zeros_like(g)
    for _ in range(50):
        av = ng.apply_a(kc, v)
        f = av + h2 * sigma[:, None] * v * v.abs() - h2 * g
        fac = TridiagonalFactor(lower, diag + 2 * h2 * sigma[:, None] * v.abs(), upper, method="thomas")
        dv = fac.solve(f)
        v = v - dv
        if dv.abs().max() <= 1e-13 * (1 + v.abs().max()):
            break
    if return_factor:
        fac = TridiagonalFactor(lower, diag + 2 * h2 * sigma[:, None] * v.abs(), upper, method="thomas")
        return v, fac
    return v


# --- сеть-колонка -----------------------------------------------------------

class ColumnNet(nn.Module):
    """
    v = P₀(K) g + S_Kᵀ · MLP(признаки K, σ, амплитуда g, низшие моды g).
    P₀ — та же аналитическая основа, что в опыте 4 (точна на высоких частотах, где
    реакция σ v|v| почти не влияет); MLP отвечает за низшие K мод — там и
    перемешивание по всей высоте, и нелинейная химия.
    """

    def __init__(self, modes=MODES, c_modes=16, width=512) -> None:
        super().__init__()
        self.modes, self.c_modes = modes, c_modes
        self.mlp = nn.Sequential(
            nn.Linear(c_modes + 4 + modes, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, modes))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, kc, sigma, g):
        n = g.shape[1]
        fwd, inv, x = ng.sine_basis(n, self.modes, g.device, g.dtype)
        kk = fwd.shape[0]
        cm = kc.mean(dim=1, keepdim=True)
        logc = torch.log(0.5 * (kc[:, :-1] + kc[:, 1:]) / cm)
        j = torch.arange(0, self.c_modes, device=g.device, dtype=g.dtype)
        feats = logc @ torch.cos(math.pi * j[None, :] * x[:, None]) * (1.0 / (n + 1))
        s = g.pow(2).mean(dim=1, keepdim=True).add(1e-12).sqrt()
        coef = torch.zeros(g.shape[0], self.modes, device=g.device, dtype=g.dtype)
        coef[:, :kk] = (g @ fwd.T) / s
        strength = sigma[:, None] * s / (math.pi ** 2 * cm) / (math.pi ** 2 * cm)  # ≈ σ·v / (π² K)
        inp = torch.cat([feats, torch.log(cm), torch.log1p(sigma[:, None]), torch.log(s),
                         torch.log1p(strength), coef], dim=1)
        corr = self.mlp(inp)[:, :kk] * s / cm / math.pi ** 2
        return ng.LineHybrid.prior(kc, g) + corr @ inv.T


def thomas_diff(lower, diag, upper, rhs):
    """Прогонка без операций «на месте» — через неё работают autograd и torch.func.jvp."""
    n = rhs.shape[1]
    cp, dp = [], []
    c_prev = upper[:, 0] / diag[:, 0]
    d_prev = rhs[:, 0] / diag[:, 0]
    cp.append(c_prev)
    dp.append(d_prev)
    for i in range(1, n):
        den = diag[:, i] - lower[:, i] * c_prev
        c_prev = upper[:, i] / den
        d_prev = (rhs[:, i] - lower[:, i] * d_prev) / den
        cp.append(c_prev)
        dp.append(d_prev)
    x = [dp[-1]]
    for i in range(n - 2, -1, -1):
        x.append(dp[i] - cp[i] * x[-1])
    return torch.stack(x[::-1], dim=1)


class ColumnNetMu(nn.Module):
    """
    Сеть-колонка v2: сеть предсказывает только эффективную скорость реакции
    μ(z) ≈ σ|v(z)| ≥ 0, а концентрацию даёт ОДНА прогонка
        (A + h² μ) v = h² g.
    Если μ угадана точно, v точное (это неподвижная точка уравнения колонки).
    Высокие частоты и краевые условия обеспечивает прогонка, сеть отвечает
    лишь за гладкий профиль μ (первые K синус-мод).
    Вход сети: признаки K, σ и «линейное» решение v₀ = A⁻¹h²g (ещё одна прогонка).
    Стоимость: 2 прогонки вместо 5–8 шагов Ньютона.
    """

    def __init__(self, modes=MODES, c_modes=16, width=512) -> None:
        super().__init__()
        self.modes, self.c_modes = modes, c_modes
        self.mlp = nn.Sequential(
            nn.Linear(c_modes + 4 + modes, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, modes + 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, kc, sigma, g):
        n = g.shape[1]
        h2 = 1.0 / (n + 1) ** 2
        fwd, inv, x = ng.sine_basis(n, self.modes, g.device, g.dtype)
        kk = fwd.shape[0]
        lower, diag, upper = ng.tridiag(kc)
        v0 = thomas_diff(lower, diag, upper, h2 * g)
        cm = kc.mean(dim=1, keepdim=True)
        logc = torch.log(0.5 * (kc[:, :-1] + kc[:, 1:]) / cm)
        j = torch.arange(0, self.c_modes, device=g.device, dtype=g.dtype)
        feats = logc @ torch.cos(math.pi * j[None, :] * x[:, None]) * (1.0 / (n + 1))
        s0 = v0.pow(2).mean(dim=1, keepdim=True).add(1e-30).sqrt()
        coef = torch.zeros(g.shape[0], self.modes, device=g.device, dtype=g.dtype)
        coef[:, :kk] = (v0 @ fwd.T) / s0
        strength = sigma[:, None] * s0 / (math.pi ** 2 * cm)        # ≈ σ|v| / (π² K): сила химии
        inp = torch.cat([feats, torch.log(cm), torch.log1p(sigma[:, None]), torch.log(s0),
                         torch.log1p(strength), coef], dim=1)
        out = self.mlp(inp)
        prof = out[:, :1] + out[:, 1:kk + 1] @ inv.T                # гладкий профиль
        # μ = σ · |v| · ρ, где |v| оценивается как |v₀|·exp(prof): ρ задаёт сеть.
        mu = sigma[:, None] * s0 * nn.functional.softplus(prof + math.log(math.e - 1)) \
            * (v0.abs() / s0 + 1e-3) / (1 + strength)
        return thomas_diff(lower, diag + h2 * mu, upper, h2 * g)


def sample_columns(batch, n, gen, balanced: bool = True):
    kc, g, _, _ = ng.sample(batch, n, gen)
    g = g / g.pow(2).mean(1, keepdim=True).sqrt()
    amp = 10 ** (-1 + 4 * torch.rand(batch, 1, generator=gen, dtype=torch.float64))   # 0.1 … 1000
    g = g * amp
    if not balanced:   # v2a: σ независимо от амплитуды — почти все колонки слабо нелинейны
        sigma = 10 ** (-1 + 3.5 * torch.rand(batch, generator=gen, dtype=torch.float64))
    else:
        # Атмосферные профили K: слабое перемешивание у земли, максимум в пограничном слое.
        atm = torch.rand(batch, generator=gen) < 0.5
        zh = (torch.arange(0, n + 1, dtype=torch.float64) + 0.5) / (n + 1)
        base = 10 ** (-2 + 1.5 * torch.rand(batch, 1, generator=gen, dtype=torch.float64))
        peak = 10 ** (-0.5 + 1.5 * torch.rand(batch, 1, generator=gen, dtype=torch.float64))
        decay = 2 + 6 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
        k_atm = base + peak * decay * zh * torch.exp(-decay * zh)
        kc = torch.where(atm[:, None], k_atm, kc)
        # Сила химии σ|v|/(π²K) равномерно в логарифме (оценка 0.01 … 1000, реальная до ~50) (оценка |v| по линейному решению).
        v0 = ng.exact_line(kc, g, torch.zeros(batch, dtype=torch.float64), torch.zeros(batch, dtype=torch.float64))
        s0 = v0.pow(2).mean(1).sqrt()
        strength = 10 ** (-2 + 5.0 * torch.rand(batch, generator=gen, dtype=torch.float64))   # до 1000 → реально до ~50
        sigma = strength * math.pi ** 2 * kc.mean(1) / s0
    sigma = torch.where(torch.rand(batch, generator=gen) < 0.1, torch.zeros_like(sigma), sigma)
    return kc, sigma, g


def make_net():
    return ColumnNetMu() if VERSION == "mu" else ColumnNet()


def train():
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=STEPS, pct_start=0.05)
    gen = torch.Generator().manual_seed(1)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        n = int(torch.randint(24, 97, (1,), generator=gen))
        kc, sigma, g = sample_columns(BATCH, n, gen)
        v_true = exact_column(kc, sigma, g).float()
        kf = kc.float()
        v = net(kf, sigma.float(), g.float())
        loss = ng.rel(v, v_true).mean() + ng.rel(ng.apply_a(kf, v), ng.apply_a(kf, v_true)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"   шаг {step:5d}: ошибка {loss.item():.2%}, {time.perf_counter() - t0:4.0f} с", flush=True)
    torch.save(net.state_dict(), WEIGHTS)
    return net


# --- двумерный разрез атмосферы ---------------------------------------------

def section(n_grid: int, sigma0: float, seed: int = 0):
    m = n_grid - 2
    h = 1.0 / (n_grid - 1)
    nd = torch.arange(1, m + 1, dtype=torch.float64) * h
    hf = (torch.arange(0, m + 1, dtype=torch.float64) + 0.5) * h
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(4, generator=g, dtype=torch.float64)

    def k_h(x, z):   # горизонтальное перемешивание (ветер, турбулентность)
        return 0.5 * torch.exp(0.3 * a[0] * torch.sin(2 * math.pi * x) + 0.3 * z)

    def k_v(x, z):   # вертикальное: сильное в пограничном слое, слабое выше (инверсия)
        return 0.05 + 2.0 * z * torch.exp(-4 * z) * (1 + 0.4 * torch.sin(math.pi * x + a[1])) * 4

    kx = k_h(hf[None, :], nd[:, None])           # [z, x½]
    kz = k_v(nd[:, None], hf[None, :])           # [x, z½]
    sigma = sigma0 * (1 + 0.6 * torch.sin(math.pi * nd))   # по x: «освещённость»
    X, Z = nd[None, :], nd[:, None]
    plume = 400 * torch.exp(-((X - 0.3) ** 2 + (Z - 0.08) ** 2) / (2 * 0.04 ** 2))
    area = 30 * torch.exp(-Z / 0.05) * (X > 0.55) * (X < 0.9)
    s = plume + area                              # [z, x]
    return dict(kx=kx, ky=kz, sigma=sigma, s=s, m=m, h=h)


def scheme_residual(p, phi):
    h2 = p["h"] ** 2
    rx = ng.apply_a(p["kx"], phi) / h2                       # строки (x)
    rz = ng.apply_a(p["ky"], phi.T).T / h2                   # колонки (z)
    r = rx + rz + p["sigma"][None, :] * phi * phi.abs() - p["s"]
    return (r.norm() / p["s"].norm()).item()


def coordinate(p, column, exact: bool, tol=1e-10, newton_max=20):
    kx, kz, sig, s = p["kx"], p["ky"], p["sigma"], p["s"]
    zero = torch.zeros(p["m"], dtype=torch.float64)
    pre = ng.Precond(p)

    def u_of(q):
        return ng.exact_line(kx, -q, zero, zero)

    def v_of(q):
        return column(kz, sig, (s + q).T).T

    q = torch.zeros(p["m"], p["m"], dtype=torch.float64)
    r = u_of(q) - v_of(q)
    r0 = r.norm().item()
    inner, newton = 0, 0
    calls = [0]
    for newton in range(1, newton_max + 1):
        if r.norm().item() < tol * r0:
            newton -= 1
            break
        if exact:
            _, fac = exact_column(kz, sig, (s + q).T, return_factor=True)
            h2 = p["h"] ** 2

            def apply_m(d, fac=fac, h2=h2):
                calls[0] += 1
                return -(u_of(d) - fac.solve(h2 * d.T).T)
        else:
            def apply_m(d, q=q):
                calls[0] += 1
                return -torch.func.jvp(lambda t: u_of(t) - v_of(t), (q,), (d,))[1]

        class Lin:
            tridiagonal_solves = 0

            def apply_m(self, d):
                return apply_m(d)

            def residual(self, d, r=r):
                return r - apply_m(d)

        res = gmres(Lin(), torch.zeros_like(q), rtol=1e-6, max_iter=500, restart=500, preconditioner=pre)
        inner += res.iterations
        # Шаг с возвратом: если невязка не уменьшилась, шаг делим пополам.
        step = 1.0
        for _ in range(8):
            q_try = q + step * res.q
            r_try = u_of(q_try) - v_of(q_try)
            if r_try.norm() < r.norm():
                break
            step *= 0.5
        q, r = q_try, r_try
    u, v = u_of(q), v_of(q)
    return 0.5 * (u + v), newton, inner, r.norm().item() / r0


def main() -> None:
    if os.path.exists(WEIGHTS):
        net = make_net()
        net.load_state_dict(torch.load(WEIGHTS))
        print(f"Сеть-колонка загружена: {WEIGHTS}")
    else:
        print(f"Обучение сети-колонки: {STEPS} шагов по {BATCH} колонок длиной 24–96")
        net = train()
    net = net.double().eval()
    for prm in net.parameters():
        prm.requires_grad_(False)

    print("\nКолонка отдельно (новые случайные колонки):")
    gen = torch.Generator().manual_seed(5)
    for n in (48, 98, 198):
        kc, sigma, g = sample_columns(1024, n, gen)
        t0 = time.perf_counter()
        v_true = exact_column(kc, sigma, g)
        t_ex = time.perf_counter() - t0
        t0 = time.perf_counter()
        v = net(kc, sigma, g)
        t_nn = time.perf_counter() - t0
        print(f"   длина {n:3d}: ошибка {ng.rel(v, v_true).mean():.2%}  "
              f"(Ньютон {t_ex * 1e3:.0f} мс, сеть {t_nn * 1e3:.0f} мс на 1024 колонки)", flush=True)

    print("\n2D-разрез: строки — прогонка, колонки с химией — СЕТЬ (без дообучения)")
    print(f"   {'сетка':>9} | {'σ₀':>5} | {'вклад химии':>11} | {'ошибка φ':>8} | {'Ньютон':>6} | {'GMRES':>5} | "
          f"{'время сеть':>10} | {'время точно':>11} | {'невязка эталона':>15}")
    for n_grid in (50, 100, 200):
        for sigma0 in (0.0, 5.0, 50.0, 300.0):
            p = section(n_grid, sigma0)
            t0 = time.perf_counter()
            phi_ref, nw_ref, in_ref, _ = coordinate(p, exact_column, exact=True)
            t_ref = time.perf_counter() - t0
            chk = scheme_residual(p, phi_ref)
            t0 = time.perf_counter()
            phi, nw, inn, _ = coordinate(p, net, exact=False)
            t_nn = time.perf_counter() - t0
            err = (phi - phi_ref).norm().item() / phi_ref.norm().item()
            chem = (p["sigma"][None, :] * phi_ref * phi_ref.abs()).norm().item() / p["s"].norm().item()
            print(f"   {n_grid:>4}x{n_grid:<4} | {sigma0:5.0f} | {chem:11.0%} | {err:8.2%} | {nw:6d} | {inn:5d} | "
                  f"{t_nn:9.1f}с | {t_ref:10.1f}с | {chk:.0e} (Ньютон {nw_ref})", flush=True)
            if n_grid == 100:
                torch.save({"phi": phi, "phi_ref": phi_ref, "s": p["s"], "sigma0": sigma0},
                           f"results/nn_column_chem_section_{int(sigma0)}.pt")


if __name__ == "__main__":
    main()
