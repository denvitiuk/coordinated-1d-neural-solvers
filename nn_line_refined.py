"""
1D-сеть v3: та же гибридная сеть, но с уточнением по невязке, и обучена именно для него.

    w₀ = N(c) f,   w_{j+1} = w_j + N(c) (f − A(c) w_j / h²),   j < R

A — трёхдиагональный оператор −(c w′)′ (его только УМНОЖАЮТ, не решают), N — сеть
из опыта 4 (LineHybrid). Отображение остаётся линейным по f, поэтому координация —
всё ещё одно линейное решение. Сеть v2 (results/nn_line_hybrid.pt) с уточнением
расходилась при перепаде k вдоль линии > 50 (I − N A не сжимает). Здесь сеть учится
на ошибке ПОСЛЕ R уточнений, поэтому учится быть сжимающей.

Дообучение стартует с весов v2. Веса: results/nn_line_refined.pt.

    python nn_line_refined.py           # обучение + проверка по перепадам
"""

from __future__ import annotations

import os
import time

import torch
from torch import nn

import nn_coordination_grids as ng

WEIGHTS = "results/nn_line_refined.pt"
R_TRAIN = 2                 # уточнений при обучении
R_USE = 2                   # уточнений при применении
STEPS = 30000
LR = 3e-4
CONTRAST_BOOST = (0.8, 1.3)  # log c умножается на случайный множитель: больше линий с большим перепадом


class LineRefined(nn.Module):
    def __init__(self, refine: int = R_USE) -> None:
        super().__init__()
        self.net = ng.LineHybrid()
        self.refine = refine

    def zero_bc(self, c, f, refine: int | None = None):
        h2 = 1.0 / (f.shape[1] + 1) ** 2
        w = self.net.zero_bc(c, f)
        for _ in range(self.refine if refine is None else refine):
            w = w + self.net.zero_bc(c, f - ng.apply_a(c, w) / h2)
        return w

    def forward(self, c, f, a, b):
        return ng.homogeneous(c, a, b) + self.zero_bc(c, f)


def sample(batch, n, gen):
    c, f, a, b = ng.sample(batch, n, gen)
    s = CONTRAST_BOOST[0] + (CONTRAST_BOOST[1] - CONTRAST_BOOST[0]) * torch.rand(batch, 1, generator=gen,
                                                                                 dtype=torch.float64)
    return torch.exp(torch.log(c) * s), f, a, b


def train():
    model = LineRefined(R_TRAIN)
    model.net.load_state_dict(torch.load(ng.WEIGHTS))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
    gen = torch.Generator().manual_seed(11)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        n = int(torch.randint(ng.TRAIN_N[0], ng.TRAIN_N[1] + 1, (1,), generator=gen))
        c, f, a, b = sample(ng.BATCH, n, gen)
        w_true = (ng.exact_line(c, f, a, b) - ng.homogeneous(c, a, b)).float()
        cf, ff = c.float(), f.float()
        loss = 0.0
        # ошибка после каждого уточнения: и само приближение, и шаг уточнения должны быть хорошими
        for r in range(R_TRAIN + 1):
            w = model.zero_bc(cf, ff, refine=r)
            # log(1 + e): редкие «взрывающиеся» линии не забивают градиент (первая попытка с
            # обычным средним не сходилась: потеря 193 → 11 → 31)
            term = (torch.log1p(ng.rel(w, w_true)).mean()
                    + torch.log1p(ng.rel(ng.apply_a(cf, w), ng.apply_a(cf, w_true))).mean())
            loss = loss + term * (2.0 if r == R_TRAIN else 0.5)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"   шаг {step:5d}: потеря {loss.item():.4f}, {time.perf_counter() - t0:4.0f} с", flush=True)
    torch.save(model.net.state_dict(), WEIGHTS)
    print(f"   обучение: {time.perf_counter() - t0:.0f} с")


def load(refine: int = R_USE) -> LineRefined:
    model = LineRefined(refine)
    model.net.load_state_dict(torch.load(WEIGHTS))
    model = model.double().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


def report():
    old = ng.LineHybrid()
    old.load_state_dict(torch.load(ng.WEIGHTS))
    old = old.double().eval()
    new = load()
    gen = torch.Generator().manual_seed(3)
    bands = ((1, 5), (5, 20), (20, 50), (50, 200), (200, 1e9))
    print("\nОшибка 1D (отн. L2), случайные f, по перепаду c вдоль линии:")
    print(f"{'':22}" + "".join(f"{f'{lo:g}–{hi:g}' if hi < 1e9 else f'>{lo:g}':>10}" for lo, hi in bands))
    for n in (48, 96, 198):
        c, f, _, _ = ng.sample(20000, n, gen)       # распределение БЕЗ усиления — как у v2
        tc = c.max(1).values / c.min(1).values
        z = torch.zeros(20000, dtype=torch.float64)
        ex = ng.exact_line(c, f, z, z)
        for name, fn in (("v2", lambda: old.zero_bc(c, f)),
                         ("v3, R=0", lambda: new.zero_bc(c, f, refine=0)),
                         (f"v3, R={R_USE}", lambda: new.zero_bc(c, f))):
            with torch.no_grad():
                e = ng.rel(fn(), ex)
            cells = "".join(f"{e[(tc >= lo) & (tc < hi)].mean():10.2%}" for lo, hi in bands)
            print(f"n={n:3} {name:14}{cells}", flush=True)


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS):
        print(f"Дообучение v3 ({STEPS} шагов, R = {R_TRAIN}):")
        train()
    report()
