"""
Сравнение с соперником: двумерная нейросеть (FNO 2D), обученная на 2D-решениях,
против координации с одномерной сетью из опыта 4 (results/nn_line_hybrid.pt).

Задача та же: −∇·(k∇φ) = 0 в [0,1]², k(x, y) — случайное гладкое поле, φ задана
на границе. FNO 2D учится на N_TRAIN решениях на сетке 50 × 50 (их сначала надо
посчитать — это входит в её стоимость), с бюджетом обучения вчетверо больше,
чем у одномерной сети (15 мин против ≈ 4 мин CPU).

Проверка:
  A. новые k и границы того же вида, сетка 50²       — родное поле FNO;
  B. то же на сетках 100² и 200²                     — перенос на другую сетку;
  C. границы другого вида (резкие ступеньки), 50²    — перенос на новые условия.

    python baseline_fno2d.py
"""

from __future__ import annotations

import math
import os
import time

import torch
from torch import nn

import nn_coordination_grids as ng

N_TRAIN = 1000
TRAIN_SECONDS = 900          # вчетверо больше, чем у 1D-сети (≈ 220 с): сравнение в пользу FNO
GRID = 50
LINE = "refined"             # "hybrid" — сеть v2 (опыт 4); "refined" — v3 с уточнением (nn_line_refined.py)
WEIGHTS = "results/baseline_fno2d.pt"
DATA = "results/baseline_fno2d_data.pt"
torch.manual_seed(0)


# --- задачи --------------------------------------------------------------

def random_problem(n_grid: int, gen: torch.Generator, bc_kind: str = "smooth"):
    a = torch.randn(6, generator=gen, dtype=torch.float64)

    def k(x, y):
        return torch.exp(0.6 * (a[0] * torch.sin(math.pi * x) * torch.sin(math.pi * y)
                                + a[1] * torch.cos(2 * math.pi * x) + a[2] * torch.cos(2 * math.pi * y)
                                + 0.5 * a[3] * torch.sin(3 * math.pi * x + a[4]) * torch.cos(math.pi * y + a[5])))

    m = n_grid - 2
    h = 1.0 / (n_grid - 1)
    nd = torch.arange(1, m + 1, dtype=torch.float64) * h
    hf = (torch.arange(0, m + 1, dtype=torch.float64) + 0.5) * h
    full = torch.arange(0, n_grid, dtype=torch.float64) * h

    def side(t, allow_zero=True):
        """Значения на стороне как функция параметра t ∈ [0, 1]."""
        if bc_kind == "smooth":
            if allow_zero and torch.rand(1, generator=gen).item() < 0.4:
                return torch.zeros_like(t)
            c = torch.randn(3, generator=gen, dtype=torch.float64)
            return (c[0] * torch.sin(math.pi * t) + 0.5 * c[1] * torch.sin(2 * math.pi * t)
                    + 0.3 * c[2] * torch.sin(3 * math.pi * t))
        # «ступенька»: резкий скачок значения посередине стороны (в обучении такого не было)
        pos = 0.3 + 0.4 * torch.rand(1, generator=gen).item()
        return (t > pos).to(t.dtype) * torch.randn(1, generator=gen, dtype=torch.float64)

    left, right, bottom, top = side(full), side(full), side(full), side(full)
    if max(t.abs().max().item() for t in (left, right, bottom, top)) == 0.0:
        top = side(full, allow_zero=False)       # хотя бы одна сторона ненулевая (иначе φ ≡ 0)
    p = dict(kx=k(hf[None, :], nd[:, None]), ky=k(nd[:, None], hf[None, :]),
             left=left[1:-1], right=right[1:-1], bottom=bottom[1:-1], top=top[1:-1], h=h, m=m)
    # Входы FNO: поле log k в узлах, граничные значения на рамке, маска рамки.
    kn = k(full[None, :], full[:, None])
    frame = torch.zeros(n_grid, n_grid, dtype=torch.float64)
    frame[:, 0], frame[:, -1], frame[0, :], frame[-1, :] = left, right, bottom, top
    mask = torch.zeros_like(frame)
    mask[:, 0] = mask[:, -1] = mask[0, :] = mask[-1, :] = 1.0
    p["fno_in"] = torch.stack([torch.log(kn), frame, mask])
    return p


def reference(p):
    phi, _, _ = ng.coordinate(ng.exact_line, p, torch.float64, tol=1e-11, newton_max=1, rtol_lin=1e-12)
    return phi


# --- FNO 2D -------------------------------------------------------------

class Spectral2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (width * width)
        self.w1 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, c, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = self.modes
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(h, w))


class FNO2d(nn.Module):
    """Стандартный FNO (Li et al., 2021): 4 спектральных слоя, дополнение до непериодичности."""

    def __init__(self, width=32, modes=12, layers=4, pad=8):
        super().__init__()
        self.pad = pad
        self.lift = nn.Conv2d(5, width, 1)
        self.spec = nn.ModuleList(Spectral2d(width, modes) for _ in range(layers))
        self.lin = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, inp):
        b, _, n, _ = inp.shape
        t = torch.linspace(0, 1, n, device=inp.device)
        gx = t[None, None, None, :].expand(b, 1, n, n)
        gy = t[None, None, :, None].expand(b, 1, n, n)
        x = self.lift(torch.cat([inp, gx, gy], dim=1))
        x = nn.functional.pad(x, (0, self.pad, 0, self.pad))
        for i, (sp, li) in enumerate(zip(self.spec, self.lin)):
            x = sp(x) + li(x)
            if i < len(self.spec) - 1:
                x = nn.functional.gelu(x)
        x = x[..., :n, :n]
        return self.proj(x)[:, 0, 1:-1, 1:-1]      # φ во внутренних узлах


def fno_resampled(fno, p, n_grid):
    """Честный вариант FNO на другой сетке: вход сжимаем до 50² (как в обучении), ответ растягиваем
    обратно билинейно. Без этого FNO с рамкой шириной в одну клетку ломается на других сетках."""
    f = nn.functional
    x = p["fno_in"].float()[None]
    x50 = f.interpolate(x[:, :2], size=(GRID, GRID), mode="bilinear", align_corners=True)
    mask = torch.zeros(1, 1, GRID, GRID)
    mask[..., 0, :] = mask[..., -1, :] = mask[..., :, 0] = mask[..., :, -1] = 1.0
    with torch.no_grad():
        inner = fno(torch.cat([x50, mask], 1))
    full = x50[:, 1:2].clone()
    full[..., 1:-1, 1:-1] = inner
    up = f.interpolate(full, size=(n_grid, n_grid), mode="bilinear", align_corners=True)
    return up[0, 0, 1:-1, 1:-1].double()


def rel(a, b):
    return torch.linalg.vector_norm((a - b).flatten(1), dim=1) / torch.linalg.vector_norm(b.flatten(1), dim=1)


# --- запуск -------------------------------------------------------------

def make_data():
    if os.path.exists(DATA):
        d = torch.load(DATA)
        print(f"Данные загружены: {DATA} ({d['x'].shape[0]} решений, генерация заняла {d['seconds']:.0f} с)")
        return d
    gen = torch.Generator().manual_seed(100)
    xs, ys = [], []
    t0 = time.perf_counter()
    for _ in range(N_TRAIN):
        p = random_problem(GRID, gen)
        xs.append(p["fno_in"].float())
        ys.append(reference(p).float())
    d = dict(x=torch.stack(xs), y=torch.stack(ys), seconds=time.perf_counter() - t0)
    torch.save(d, DATA)
    print(f"Сгенерировано {N_TRAIN} 2D-решений {GRID}×{GRID} за {d['seconds']:.0f} с")
    return d


def train(d):
    net = FNO2d()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(1)
    t0 = time.perf_counter()
    step = 0
    while time.perf_counter() - t0 < TRAIN_SECONDS:
        lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * min(1.0, (time.perf_counter() - t0) / TRAIN_SECONDS)))
        for g in opt.param_groups:
            g["lr"] = max(lr, 1e-5)
        idx = torch.randint(0, d["x"].shape[0], (20,), generator=gen)
        loss = rel(net(d["x"][idx]), d["y"][idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 500 == 0:
            print(f"   шаг {step:5d}: ошибка {loss.item():.2%}, {time.perf_counter() - t0:4.0f} с", flush=True)
    torch.save(net.state_dict(), WEIGHTS)
    print(f"   обучение: {step} шагов за {time.perf_counter() - t0:.0f} с")
    return net


def main() -> None:
    d = make_data()
    if os.path.exists(WEIGHTS):
        fno = FNO2d()
        fno.load_state_dict(torch.load(WEIGHTS))
    else:
        print(f"Обучение FNO 2D ({TRAIN_SECONDS} с):")
        fno = train(d)
    fno.eval()
    print(f"Параметров FNO 2D: {sum(p.numel() for p in fno.parameters()):,}")

    if LINE == "refined":
        import nn_line_refined
        line = nn_line_refined.load()
    else:
        line = ng.LineHybrid()
        line.load_state_dict(torch.load(ng.WEIGHTS))
        line = line.double().eval()
        for prm in line.parameters():
            prm.requires_grad_(False)
    print(f"1D-сеть: {LINE}")

    cases = [("A. новые k, гладкие границы", 50, "smooth"),
             ("B. другая сетка", 100, "smooth"),
             ("B. другая сетка", 200, "smooth"),
             ("C. границы-ступеньки", 50, "step"),
             ("C. границы-ступеньки", 100, "step")]
    print("(на сетках, отличных от 50², FNO работает через пересэмплинг 50² ↔ n², см. fno_resampled)")
    print(f"\n{'проверка':28} | {'сетка':>7} | {'FNO 2D':>16} | {'координация + 1D-сеть':>22}")
    gen = torch.Generator().manual_seed(555)
    summary = []
    for name, n_grid, kind in cases:
        e_fno, e_crd, t_fno, t_crd = [], [], 0.0, 0.0
        for _ in range(20):
            p = random_problem(n_grid, gen, kind)
            ref = reference(p)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = fno(p["fno_in"].float()[None])[0].double()
            t_fno += time.perf_counter() - t0
            if n_grid != GRID:
                out = fno_resampled(fno, p, n_grid)
            t0 = time.perf_counter()
            phi, _, _ = ng.coordinate(line, p, torch.float64, tol=1e-10, newton_max=1, rtol_lin=1e-10)
            t_crd += time.perf_counter() - t0
            e_fno.append(((out - ref).norm() / ref.norm()).item())
            e_crd.append(((phi - ref).norm() / ref.norm()).item())
        mf, mc = sum(e_fno) / len(e_fno), sum(e_crd) / len(e_crd)
        summary.append((name, n_grid, mf, max(e_fno), mc, max(e_crd)))
        print(f"{name:28} | {n_grid:>3}×{n_grid:<3} | {mf:6.2%} (макс {max(e_fno):6.1%}) | "
              f"{mc:6.2%} (макс {max(e_crd):5.1%})   "
              f"[время на задачу: FNO {t_fno / 20 * 1e3:.0f} мс, коорд. {t_crd / 20 * 1e3:.0f} мс]", flush=True)
    torch.save(summary, "results/baseline_fno2d_summary.pt")
    print(f"\nСтоимость подготовки: FNO 2D — {d['seconds']:.0f} с на данные (1000 двумерных решений) + "
          f"{TRAIN_SECONDS} с обучения; 1D-сеть — данные бесплатны (прогонка), ≈ 220 с обучения.")


if __name__ == "__main__":
    main()
