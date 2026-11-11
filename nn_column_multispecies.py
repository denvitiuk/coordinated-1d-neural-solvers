"""
Опыт 7: колонка атмосферы с НЕСКОЛЬКИМИ веществами (цикл NO–NO₂–O₃) — нейросеть,
горизонтальный перенос — прогонка, связь — координация.

Вещества c = (a, b, o) = (NO, NO₂, O₃). Реакции (на пальцах):
    солнце разбивает NO₂:     NO₂ → NO + O₃          скорость J·b
    NO «съедает» озон:        NO + O₃ → NO₂          скорость k·a·o   (нелинейно!)
    NO₂ уходит в кислоту:     NO₂ → (вымывание)      скорость λ·b
    озон оседает / распадается                       скорость δ·o
Уравнения (установившиеся):
    −∇·(K∇a) + k a o − J b             = s_a
    −∇·(K∇b) − k a o + (J + λ) b       = s_b
    −∇·(K∇o) + k a o − J b + δ o       = s_o
λ и δ ломают оба «закона сохранения» (NO + NO₂ и O₃ + NO₂), поэтому три вещества
действительно связаны и к одному уравнению не сводятся.

Точная колонка: Ньютон, на каждом шаге блочная прогонка с блоками 3 × 3.
Сеть-колонка: угадывает только профиль озона ô(z); с ним член k a o становится
линейным (k ô · a) и колонка решается ОДНОЙ блочной прогонкой. Если ô точен —
ответ точный. Затем NEWTON_STEPS шагов Ньютона от этой точки (каждый — одна блочная
прогонка): сеть — «умная стартовая точка», ответ доводится до точного.

Почему не Пикар (ô ← o): при сильном «съедании» озона (NO ≫ O₃, у трубы и дорог)
уточнение по Пикару расходится (среднее 1400 %); раздельная заморозка (k ô·a в
уравнениях NO/NO₂, k â·o в уравнении O₃) сходится, но медленно (среднее 5–9 %).
Ньютон от точки сети: 2 шага — медиана 1e-14, 3 шага — худшая колонка 1e-4.

    python nn_column_multispecies.py
"""

from __future__ import annotations

import math
import os
import time

import torch
from torch import nn

import nn_coordination_grids as ng
from coordinator_laplace_descent import gmres

WEIGHTS = "results/nn_column_multi.pt"
DATA = "results/nn_column_multi_data.pt"
STEPS = 12000
BATCH = 128
MODES = 24
REFINE = 2                     # шагов Пикара в функции потерь при обучении (так обучена сеть)
NEWTON_STEPS = 3               # шагов Ньютона от точки сети при применении
TRAIN_N = (24, 32, 48, 64)     # длины колонок в обучающих данных
torch.manual_seed(0)


# --- химия ------------------------------------------------------------------

def chem(c, prm):
    """Члены химии R(c) (как «потери»: уравнение A c / h² + R(c) = g) и их Якобиан.
    c [B, n, 3]; prm — dict J, k, lam, dep формы [B]."""
    a, b, o = c[..., 0], c[..., 1], c[..., 2]
    J, k, lam, dep = (prm[s][:, None] for s in ("J", "k", "lam", "dep"))
    r2, r1 = k * a * o, J * b
    R = torch.stack([r2 - r1, r1 - r2 + lam * b, r2 - r1 + dep * o], dim=-1)
    z = torch.zeros_like(a)
    jac = torch.stack([
        torch.stack([k * o, -J + z, k * a], -1),
        torch.stack([-k * o, J + lam + z, -k * a], -1),
        torch.stack([k * o, -J + z, k * a + dep], -1)], dim=-2)          # [B, n, 3, 3]
    return R, jac


def block_thomas(lower, diag_s, blocks, upper, rhs):
    """Блочная прогонка: (диаг. диффузия · I + blocks) x = rhs, внедиагональные блоки = скаляр · I.
    lower/upper/diag_s [B, n], blocks [B, n, 3, 3], rhs [B, n, 3]. Без операций «на месте»
    (работают autograd и torch.func.jvp)."""
    n = rhs.shape[1]
    eye = torch.eye(3, dtype=rhs.dtype, device=rhs.device)
    cp, dp = [], []
    c_prev = d_prev = None
    for i in range(n):
        d = diag_s[:, i, None, None] * eye + blocks[:, i]
        r = rhs[:, i]
        if i > 0:
            d = d - lower[:, i, None, None] * c_prev
            r = r - lower[:, i, None] * d_prev
        inv = torch.linalg.inv(d)
        c_prev = inv * upper[:, i, None, None]                         # D⁻¹ · (u I)
        d_prev = (inv @ r[..., None])[..., 0]
        cp.append(c_prev)
        dp.append(d_prev)
    x = [dp[-1]]
    for i in range(n - 2, -1, -1):
        x.append(dp[i] - (cp[i] @ x[-1][..., None])[..., 0])
    return torch.stack(x[::-1], dim=1)


def apply_a3(kc, c):
    return torch.stack([ng.apply_a(kc, c[..., s]) for s in range(3)], dim=-1)


def exact_column(kc, prm, g, return_jac=False):
    """Ньютон: A c + h² R(c) = h² g, c = 0 на концах. kc [B, n+1], g [B, n, 3]."""
    n = g.shape[1]
    h2 = 1.0 / (n + 1) ** 2
    lower, diag, upper = ng.tridiag(kc)
    c = torch.zeros_like(g)
    for it in range(60):
        R, jac = chem(c, prm)
        F = apply_a3(kc, c) + h2 * R - h2 * g
        dc = block_thomas(lower, diag, h2 * jac, upper, F)
        step = 1.0
        if it < 3:                                   # первые шаги — осторожно (сильная химия)
            fn = F.norm()
            for _ in range(10):
                c_try = c - step * dc
                Rt, _ = chem(c_try, prm)
                if (apply_a3(kc, c_try) + h2 * Rt - h2 * g).norm() < fn:
                    break
                step *= 0.5
        c = c - step * dc
        if step == 1.0 and dc.abs().max() <= 1e-12 * (1 + c.abs().max()):
            break
    if return_jac:
        _, jac = chem(c, prm)
        return c, (lower, diag, h2 * jac, upper)
    return c


def frozen_solve(kc, prm, g, o_hat):
    """Колонка при заданном профиле озона ô: k a o → k ô a, система линейна — одна блочная прогонка."""
    n = g.shape[1]
    h2 = 1.0 / (n + 1) ** 2
    lower, diag, upper = ng.tridiag(kc)
    J, k, lam, dep = (prm[s][:, None] for s in ("J", "k", "lam", "dep"))
    ko = k * o_hat
    z = torch.zeros_like(ko)
    blocks = torch.stack([
        torch.stack([ko, -J + z, z], -1),
        torch.stack([-ko, J + lam + z, z], -1),
        torch.stack([ko, -J + z, dep + z], -1)], dim=-2)
    return block_thomas(lower, diag, h2 * blocks, upper, h2 * g)


# --- сеть-колонка ------------------------------------------------------------

class ColumnNetO3(nn.Module):
    """
    Вход: признаки K (косинус-моды log K), безразмерные скорости J, k, λ, δ и
    «линейный» ответ c₀ (без реакции NO + O₃, одна блочная прогонка) по низшим модам.
    Выход: профиль озона ô = c₀,o-подобный профиль × softplus(гладкая поправка).
    Ответ: блочная прогонка с замороженным ô, затем REFINE шагов Пикара.
    """

    def __init__(self, modes=MODES, c_modes=12, width=384, refine=REFINE) -> None:
        super().__init__()
        self.modes, self.c_modes, self.refine = modes, c_modes, refine
        n_in = c_modes + 1 + 4 + 3 + 3 * modes
        self.mlp = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, modes + 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def guess(self, kc, prm, g):
        n = g.shape[1]
        fwd, inv, x = ng.sine_basis(n, self.modes, g.device, g.dtype)
        kk = fwd.shape[0]
        zero = torch.zeros_like(prm["k"])
        c0 = frozen_solve(kc, {**prm, "k": zero}, g, torch.zeros_like(g[..., 0]))   # без NO + O₃
        cm = kc.mean(dim=1, keepdim=True)
        logc = torch.log(0.5 * (kc[:, :-1] + kc[:, 1:]) / cm)
        j = torch.arange(0, self.c_modes, device=g.device, dtype=g.dtype)
        feats = logc @ torch.cos(math.pi * j[None, :] * x[:, None]) * (1.0 / (n + 1))
        sc = c0.pow(2).mean(dim=1).add(1e-30).sqrt()                                 # [B, 3]
        coef = torch.zeros(g.shape[0], 3, self.modes, device=g.device, dtype=g.dtype)
        coef[:, :, :kk] = torch.einsum("bns,kn->bsk", c0, fwd) / sc[..., None]
        pk = math.pi ** 2 * cm
        rates = torch.stack([prm["J"], prm["k"] * sc[:, 0], prm["lam"], prm["dep"]], 1) / pk
        inp = torch.cat([feats, torch.log(cm), torch.log1p(rates), torch.log(sc), coef.flatten(1)], dim=1)
        out = self.mlp(inp)
        prof = out[:, :1] + out[:, 1:kk + 1] @ inv.T
        o_ref = c0[..., 2].abs() + 1e-3 * sc[:, 2:3]
        return o_ref * nn.functional.softplus(prof + math.log(math.e - 1))

    def forward(self, kc, prm, g, newton: int = NEWTON_STEPS):
        c = frozen_solve(kc, prm, g, self.guess(kc, prm, g))
        return newton_steps(kc, prm, g, c, newton)


def newton_steps(kc, prm, g, c, steps):
    """Чистые шаги Ньютона (без поиска шага): от хорошей стартовой точки сходятся квадратично."""
    n = g.shape[1]
    h2 = 1.0 / (n + 1) ** 2
    lower, diag, upper = ng.tridiag(kc)
    for _ in range(steps):
        R, jac = chem(c, prm)
        F = apply_a3(kc, c) + h2 * R - h2 * g
        c = c - block_thomas(lower, diag, h2 * jac, upper, F)
    return c


# --- обучающие колонки ---------------------------------------------------------

def sample_columns(batch, n, gen):
    zh = (torch.arange(0, n + 1, dtype=torch.float64) + 0.5) / (n + 1)
    z = torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)
    rnd = lambda *s: torch.rand(*s, generator=gen, dtype=torch.float64)   # noqa: E731
    # K: атмосферный профиль (слабое у земли, максимум в пограничном слое) или случайный гладкий
    base = 10 ** (-2 + 1.5 * rnd(batch, 1))
    peak = 10 ** (-0.5 + 1.5 * rnd(batch, 1))
    decay = 2 + 6 * rnd(batch, 1)
    k_atm = base + peak * decay * zh * torch.exp(-decay * zh)
    k_rand, _, _, _ = ng.sample(batch, n, gen)
    kc = torch.where((rnd(batch) < 0.7)[:, None], k_atm, k_rand * 10 ** (-1 + rnd(batch, 1)))
    # источники: NO — у земли и «шлейф» на случайной высоте; NO₂ — доля от NO; O₃ — фон
    amp = 10 ** (1 + 2 * rnd(batch, 1))
    h0 = 0.02 + 0.3 * rnd(batch, 1)
    w0 = 0.02 + 0.1 * rnd(batch, 1)
    s_no = amp * (torch.exp(-((z - h0) / w0) ** 2) + 0.3 * rnd(batch, 1) * torch.exp(-z / 0.05))
    s_no2 = 0.2 * rnd(batch, 1) * s_no
    s_o3 = 10 ** (0 + 1.5 * rnd(batch, 1)) * (0.5 + rnd(batch, 1) * z)
    g = torch.stack([s_no, s_no2, s_o3], dim=-1)
    km = kc.mean(1)
    prm = dict(J=10 ** (-1 + 2.5 * rnd(batch)) * km * math.pi ** 2 * (rnd(batch) > 0.05),
               k=10 ** (-2.5 + 4.5 * rnd(batch)) * km * math.pi ** 2 * (rnd(batch) > 0.05),
               lam=10 ** (-2 + 2 * rnd(batch)) * km * math.pi ** 2,
               dep=10 ** (-2 + 2 * rnd(batch)) * km * math.pi ** 2)
    return kc, prm, g


def make_data():
    if os.path.exists(DATA):
        return torch.load(DATA)
    gen = torch.Generator().manual_seed(1)
    data = {}
    t0 = time.perf_counter()
    for n in TRAIN_N:
        chunks = []
        for _ in range(24):
            kc, prm, g = sample_columns(512, n, gen)
            chunks.append((kc, prm, g, exact_column(kc, prm, g)))
        data[n] = (torch.cat([c[0] for c in chunks]),
                   {s: torch.cat([c[1][s] for c in chunks]) for s in ("J", "k", "lam", "dep")},
                   torch.cat([c[2] for c in chunks]), torch.cat([c[3] for c in chunks]))
        print(f"   данные n = {n}: {data[n][0].shape[0]} колонок, {time.perf_counter() - t0:.0f} с", flush=True)
    torch.save(data, DATA)
    return data


def rel3(c, c_true):
    """Средняя по веществам относительная ошибка колонки, [B]."""
    return torch.stack([ng.rel(c[..., s], c_true[..., s]) for s in range(3)], 1).mean(1)


def train(data):
    net = ColumnNetO3().double()       # данные и блочная прогонка — float64
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=STEPS, pct_start=0.05)
    gen = torch.Generator().manual_seed(2)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        n = TRAIN_N[int(torch.randint(len(TRAIN_N), (1,), generator=gen))]
        kc, prm, g, ct = data[n]
        idx = torch.randint(kc.shape[0], (BATCH,), generator=gen)
        kb, gb, cb = kc[idx], g[idx], ct[idx]
        pb = {s: v[idx] for s, v in prm.items()}
        o_hat = net.guess(kb, pb, gb)
        c = frozen_solve(kb, pb, gb, o_hat)
        loss = torch.log1p(rel3(c, cb)).mean() + torch.log1p(ng.rel(o_hat, cb[..., 2])).mean()
        for _ in range(REFINE):
            c = frozen_solve(kb, pb, gb, c[..., 2].clamp_min(0.0))
        loss = loss + 2 * torch.log1p(rel3(c, cb)).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"   шаг {step:5d}: потеря {loss.item():.4f}, {time.perf_counter() - t0:4.0f} с", flush=True)
    torch.save(net.state_dict(), WEIGHTS)
    return net


# --- двумерный разрез: город, труба, солнце ---------------------------------------

def section(n_grid: int, k0: float):
    m = n_grid - 2
    h = 1.0 / (n_grid - 1)
    nd = torch.arange(1, m + 1, dtype=torch.float64) * h
    hf = (torch.arange(0, m + 1, dtype=torch.float64) + 0.5) * h
    kx = 0.5 * torch.exp(0.3 * torch.sin(2 * math.pi * hf[None, :]) + 0.3 * nd[:, None])     # [z, x½]
    kz = 0.05 + 8.0 * hf[None, :] * torch.exp(-4 * hf[None, :]) * (1 + 0.4 * torch.sin(math.pi * nd[:, None] + 0.5))
    X, Z = nd[None, :], nd[:, None]
    plume = 400 * torch.exp(-((X - 0.3) ** 2 + (Z - 0.08) ** 2) / (2 * 0.04 ** 2))         # труба
    traffic = 60 * torch.exp(-Z / 0.04) * (X > 0.55) * (X < 0.9)                        # дороги
    s_no = plume + traffic
    s = torch.stack([s_no, 0.1 * s_no, 15.0 * (0.5 + Z).expand(m, m)])                   # [3, z, x]
    J = 20.0 * (1 + 0.5 * torch.sin(math.pi * nd))                                       # солнце по x
    prm = dict(J=J, k=k0 * torch.ones(m, dtype=torch.float64), lam=2.0 * torch.ones(m, dtype=torch.float64),
               dep=1.0 * torch.ones(m, dtype=torch.float64))
    return dict(kx=kx, ky=kz, s=s, prm=prm, m=m, h=h)


def coordinate(p, column, tol=1e-9, newton_max=25):
    """Ньютон по q. Якобиан колонки — точный, в найденном колонкой состоянии (для обоих
    вариантов одинаково), поэтому сравнение времени честное: отличается только то, как
    находится само состояние колонки (Ньютон с нуля или сеть + 3 шага Ньютона)."""
    kx, kz, s, prm, m = p["kx"], p["ky"], p["s"], p["prm"], p["m"]
    zero = torch.zeros(3 * m, dtype=torch.float64)
    kx3 = kx.repeat(3, 1)
    pre1 = ng.Precond(p)

    def pre(z):
        return torch.stack([pre1(z[i]) for i in range(3)])

    def u_of(q):
        return ng.exact_line(kx3, (-q).reshape(3 * m, m), zero, zero).reshape(3, m, m)

    state = {}
    t_col = [0.0]

    def v_of(q):
        t0 = time.perf_counter()
        c = column(kz, prm, (s + q).permute(2, 1, 0))
        t_col[0] += time.perf_counter() - t0
        state["c"] = c
        return c.permute(2, 1, 0)

    q = torch.zeros(3, m, m, dtype=torch.float64)
    r = u_of(q) - v_of(q)
    r0 = r.norm().item()
    inner = newton = 0
    for newton in range(1, newton_max + 1):
        if r.norm().item() < tol * r0:
            newton -= 1
            break
        h2 = 1.0 / (m + 1) ** 2
        lo, dg, up = ng.tridiag(kz)
        _, jac = chem(state["c"], prm)

        def apply_m(d, lo=lo, dg=dg, up=up, bl=h2 * jac, h2=h2):
            dv = block_thomas(lo, dg, bl, up, h2 * d.permute(2, 1, 0)).permute(2, 1, 0)
            return -(u_of(d) - dv)

        class Lin:
            tridiagonal_solves = 0

            def apply_m(self, d):
                return apply_m(d)

            def residual(self, d, r=r):
                return r - apply_m(d)

        res = gmres(Lin(), torch.zeros_like(q), rtol=1e-6, max_iter=300, restart=300, preconditioner=pre)
        inner += res.iterations
        step = 1.0
        for _ in range(8):
            q_try = q + step * res.q
            r_try = u_of(q_try) - v_of(q_try)
            if r_try.norm() < r.norm():
                break
            step *= 0.5
        q, r = q_try, r_try
    u, v = u_of(q), v_of(q)
    return 0.5 * (u + v), newton, inner, t_col[0]


def main() -> None:
    print("Данные: точные колонки (Ньютон, блочная прогонка 3 × 3)")
    data = make_data()
    if os.path.exists(WEIGHTS):
        net = ColumnNetO3()
        net.load_state_dict(torch.load(WEIGHTS))
        print(f"Сеть загружена: {WEIGHTS}")
    else:
        print(f"Обучение: {STEPS} шагов по {BATCH} колонок, уточнений {REFINE}")
        net = train(data)
    net = net.double().eval()
    for prm in net.parameters():
        prm.requires_grad_(False)

    print("\nКолонка отдельно (новые колонки): ошибка (медиана / 99 % / худшая) и время на 1024 колонки")
    gen = torch.Generator().manual_seed(9)
    for n in (48, 98):
        kc, prm, g = sample_columns(1024, n, gen)
        t0 = time.perf_counter()
        ct = exact_column(kc, prm, g)
        t_ex = time.perf_counter() - t0
        for k in (0, 1, 2, 3):
            t0 = time.perf_counter()
            with torch.no_grad():
                c = net(kc, prm, g, newton=k)
            t_nn = time.perf_counter() - t0
            e = rel3(c, ct)
            print(f"   n = {n:3d}, сеть + {k} шага Ньютона: {e.median():.1e} / {torch.quantile(e, 0.99):.1e} / "
                  f"{e.max():.1e};  сеть {t_nn * 1e3:4.0f} мс, точный Ньютон {t_ex * 1e3:4.0f} мс", flush=True)
        for k in (3, 5):
            e = rel3(newton_steps(kc, prm, g, torch.zeros_like(g), k), ct)
            print(f"      (Ньютон с нуля, {k} шагов: {e.median():.1e} / {torch.quantile(e, 0.99):.1e} / {e.max():.1e})")

    def net_column(kc, prm, g):
        with torch.no_grad():
            return net(kc, prm, g)

    print("\n2D-разрез: строки — прогонка, колонки с химией 3 веществ — сеть + 3 шага Ньютона")
    print(f"   {'сетка':>9} | {'k':>3} | {'вклад NO+O₃':>11} | {'ошибка NO / NO₂ / O₃':>26} | {'Ньютон':>6} | "
          f"{'GMRES':>5} | {'колонки: сеть / точно':>22} | {'всего: сеть / точно':>20}")
    for n_grid in (50, 100):
        for k0 in (0.0, 5.0, 20.0, 50.0):         # k0 = 300: расходится уже точная координация
            p = section(n_grid, k0)
            t0 = time.perf_counter()
            ref, nw_r, in_r, tc_r = coordinate(p, exact_column)
            t_ref = time.perf_counter() - t0
            t0 = time.perf_counter()
            phi, nw, inn, tc = coordinate(p, net_column)
            t_nn = time.perf_counter() - t0
            err = [((phi[i] - ref[i]).norm() / ref[i].norm()).item() for i in range(3)]
            share = (k0 * ref[0] * ref[2]).norm().item() / p["s"].norm().item()
            print(f"   {n_grid:>4}x{n_grid:<4} | {k0:3.0f} | {share:11.0%} | "
                  f"{err[0]:.1e} / {err[1]:.1e} / {err[2]:.1e} | {nw:3d}/{nw_r:<2d} | {inn:5d} | "
                  f"{tc:8.1f}с / {tc_r:6.1f}с | {t_nn:7.1f}с / {t_ref:6.1f}с", flush=True)
            if n_grid == 100:
                torch.save({"phi": phi, "ref": ref, "k0": k0, "s": p["s"]},
                           f"results/nn_column_multi_section_{int(k0)}.pt")


if __name__ == "__main__":
    main()
