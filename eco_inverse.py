"""
Обратная задача: найти источники загрязнения по измерениям станций.

Прямая модель (eco_pollution.py) линейна: c = L⁻¹ S. Показание станции m:
    d_m = c(x_m) = ⟨e_m, L⁻¹ S⟩ = ⟨φ_m, S⟩,   Lᵀ φ_m = e_m.
φ_m — «след» (footprint) станции: одна сопряжённая задача, решённая той же
координацией (Lxᵀ, Lyᵀ, ADI). M станций -> M сопряжённых задач, и дальше
вклад ЛЮБОГО источника в ЛЮБОМ месте в показания всех станций известен
без новых решений: столбец матрицы источник -> станции = ⟨φ_m, S⟩.

Это та же сопряжённая техника, что и в градиенте диплома (Aᵀp = r).

Сценарий (все числа синтетические):
  «истина» считается на мелкой сетке (401 x 201) — ТЭЦ выбрасывает 130,
  а заявляет 100; кроме того, есть незаявленный источник (пожар на складе ГСМ).
  Восстановление — на грубой сетке (201 x 101), чтобы не решать задачу
  той же моделью, которой созданы данные («inverse crime»).
  Шум измерений 5 % + 0.02.

  1. Проверка заявленных выбросов: χ² данных при заявленных мощностях.
  2. Оценка мощностей известных источников (взвешенные МНК + погрешности).
  3. Поиск незаявленного источника: перебор всех узлов сетки как кандидатов
     (векторно, по готовым следам), мощность и место + 95 %-я область.
  4. Сколько станций нужно: ошибка места/мощности от числа станций.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from eco_pollution import (
    EcoSolver,
    Region,
    Source,
    build_operators,
    default_sources,
    full_field,
    nearest_node,
)

Tensor = torch.Tensor

HIDDEN = Source("Пожар (незаявленный)", 11.0, 8.0, rate=50.0, sigma=0.3)
TRUE_RATES = {"ТЭЦ": 130.0, "Промзона": 60.0, "Автобан": 40.0}
NOISE_REL, NOISE_ABS = 0.05, 0.02
CHI2_95_2D = 5.99  # Δχ² для 95 %-й области по двум параметрам (x, y)
# Ошибка модели: грубая сетка против мелкой расходится до ~3.7 % в показаниях
# станций. В веса МНК входит √(σ_шум² + (MODEL_REL·d)²), иначе при многих станциях
# 95 %-я область получается слишком узкой (истина выпадает из неё).
MODEL_REL = 0.03
# Кандидаты ближе этого к известным точечным источникам исключаются: добавка в той же
# точке неотличима от изменения мощности самого источника (столбцы почти совпадают).
EXCLUDE_KM = 1.0


def station_pool(count: int = 24, seed: int = 1) -> list[tuple[float, float]]:
    """Станции в узлах сетки 0.1 км, внутри района, чуть больше на востоке."""
    gen = torch.Generator().manual_seed(seed)
    pts = []
    while len(pts) < count:
        x = 1.0 + 18.0 * torch.rand(1, generator=gen).sqrt().item()  # гуще к востоку
        y = 0.8 + 8.4 * torch.rand(1, generator=gen).item()
        pts.append((round(x, 1), round(y, 1)))
    return pts


def true_sources() -> list[Source]:
    return [replace(s, rate=TRUE_RATES[s.name]) for s in default_sources()] + [HIDDEN]


def clean_observations(region: Region, sources: list[Source], stations) -> Tensor:
    """Точные показания станций: решение прямой задачи на сетке region."""
    ops = build_operators(replace(region, sources=sources))
    c, _ = EcoSolver(ops).solve(ops.source)
    return torch.tensor([c[nearest_node(region, x, y)].item() for x, y in stations],
                        dtype=torch.float64)


def inversion_sigma(d: Tensor, noise_sigma: Tensor) -> Tensor:
    return torch.sqrt(noise_sigma**2 + (MODEL_REL * d.abs()) ** 2)


def add_noise(clean: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    """
    Шум 5 % + 0.02 (точность прибора). Возвращает (данные, σ для весов МНК),
    где σ включает и ошибку модели (MODEL_REL).
    """
    noise = NOISE_REL * clean.abs() + NOISE_ABS
    gen = torch.Generator().manual_seed(seed)
    d = clean + noise * torch.randn(len(clean), generator=gen, dtype=torch.float64)
    return d, inversion_sigma(d, noise)


def observe(region: Region, sources: list[Source], stations, seed: int = 7):
    """Синтетические измерения: решение на сетке region + шум."""
    return add_noise(clean_observations(region, sources, stations), seed)


@dataclass
class Footprints:
    region: Region
    phi: Tensor  # [M, ny_in, nx_in]
    seconds: float
    iterations: list[int]

    @property
    def cell(self) -> float:
        return self.region.hx * self.region.hy

    def column(self, source_field: Tensor) -> Tensor:
        """Вклад источника (поле S) в показания всех станций."""
        return torch.einsum("mij,ij->m", self.phi, source_field) * self.cell

    def candidate_columns(self, sigma: float) -> Tensor:
        """
        Вклад гауссова источника единичной мощности с центром в каждом узле:
        свёртка следов с ядром (нормированным на 1). [M, ny_in, nx_in].
        """
        r = self.region
        kx = max(1, math.ceil(3 * sigma / r.hx))
        ky = max(1, math.ceil(3 * sigma / r.hy))
        x = torch.arange(-kx, kx + 1, dtype=torch.float64) * r.hx
        y = torch.arange(-ky, ky + 1, dtype=torch.float64) * r.hy
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return F.conv2d(self.phi.unsqueeze(1), kernel[None, None], padding=(ky, kx)).squeeze(1)


def compute_footprints(region: Region, stations) -> Footprints:
    ops = build_operators(region)
    adj = EcoSolver(ops, adjoint=True)
    cell = region.hx * region.hy
    start = time.perf_counter()
    phis, its = [], []
    for x, y in stations:
        delta = torch.zeros_like(ops.source)
        delta[nearest_node(region, x, y)] = 1.0 / cell
        phi, res = adj.solve(delta)
        phis.append(phi)
        its.append(res.iterations)
    return Footprints(region, torch.stack(phis), time.perf_counter() - start, its)


@dataclass
class Fit:
    rates: Tensor
    std: Tensor
    chi2: float


def weighted_lsq(g: Tensor, d: Tensor, sigma: Tensor) -> Fit:
    a, b = g / sigma[:, None], d / sigma
    rates = torch.linalg.lstsq(a, b.unsqueeze(1)).solution.squeeze(1)
    cov = torch.linalg.inv(a.T @ a)
    chi2 = torch.sum((a @ rates - b) ** 2).item()
    return Fit(rates, torch.sqrt(torch.diag(cov)), chi2)


@dataclass
class Scan:
    misfit: Tensor  # χ² для каждого узла-кандидата [ny_in, nx_in]
    rate: Tensor  # лучшая мощность для каждого узла
    best: tuple[int, int]
    best_xy: tuple[float, float]
    area95: float  # км²


def scan_extra_source(fp: Footprints, g_known: Tensor, d: Tensor, sigma: Tensor,
                      blob_sigma: float = 0.3,
                      exclude: list[tuple[float, float]] | None = None) -> Scan:
    """
    Для каждого узла: МНК по [известные источники + источник в этом узле],
    мощность нового ≥ 0. Векторно: проектор на дополнение к известным столбцам.
    """
    w = 1.0 / sigma
    q, rk = torch.linalg.qr(g_known * w[:, None])
    b = d * w
    pb = b - q @ (q.T @ b)
    cand = fp.candidate_columns(blob_sigma)
    shape = cand.shape[1:]
    c = cand.reshape(len(d), -1) * w[:, None]
    pc = c - q @ (q.T @ c)
    norm2 = torch.sum(pc * pc, dim=0).clamp_min(1e-300)
    rate = (pc.T @ pb / norm2).clamp_min(0.0)
    misfit = torch.sum(pb * pb) - 2 * rate * (pc.T @ pb) + rate**2 * norm2
    # Мощности известных источников при этом кандидате: R k = Qᵀ(b − rate·c).
    # Отрицательный выброс физически невозможен -> такой кандидат отбрасывается.
    rhs = (q.T @ b)[:, None] - (q.T @ c) * rate[None, :]
    known = torch.linalg.solve_triangular(rk, rhs, upper=True)
    misfit = torch.where((known < 0).any(dim=0), torch.full_like(misfit, float("inf")), misfit)
    if exclude:
        r = fp.region
        xx, yy = r.grid()
        xx, yy = xx[1:-1, 1:-1].reshape(-1), yy[1:-1, 1:-1].reshape(-1)
        for px, py in exclude:
            misfit = torch.where(torch.hypot(xx - px, yy - py) < EXCLUDE_KM,
                                 torch.full_like(misfit, float("inf")), misfit)
    k = int(torch.argmin(misfit))
    i, j = divmod(k, shape[1])
    r = fp.region
    area = (misfit - misfit[k] < CHI2_95_2D).sum().item() * r.hx * r.hy
    return Scan(misfit.reshape(shape), rate.reshape(shape), (i, j),
                ((j + 1) * r.hx, (i + 1) * r.hy), area)


@dataclass
class Located:
    error_km: float
    rates: Tensor  # известные + новый
    std: Tensor
    chi2: float
    truth_in_95: bool
    area95: float


def locate(fp: Footprints, g_known: Tensor, d: Tensor, sigma: Tensor,
           cand: Tensor | None = None) -> tuple[Scan, Located]:
    """Поиск нового источника + повторный МНК в найденной точке + проверка истины."""
    sc = scan_extra_source(fp, g_known, d, sigma, HIDDEN.sigma, point_sources())
    cand = fp.candidate_columns(HIDDEN.sigma) if cand is None else cand
    col = cand[:, sc.best[0], sc.best[1]]
    fit = weighted_lsq(torch.cat([g_known, col[:, None]], dim=1), d, sigma)
    i, j = nearest_node(fp.region, HIDDEN.x, HIDDEN.y)
    inside = (sc.misfit[i, j] - sc.misfit.min()).item() < CHI2_95_2D  # nearest_node: внутренний узел
    err = math.hypot(sc.best_xy[0] - HIDDEN.x, sc.best_xy[1] - HIDDEN.y)
    return sc, Located(err, fit.rates, fit.std, fit.chi2, inside, sc.area95)


def point_sources() -> list[tuple[float, float]]:
    return [(s.x, s.y) for s in default_sources() if s.x_end is None]


def unit(sources: list[Source]) -> list[Source]:
    return [replace(s, rate=1.0) for s in sources]


def known_columns(fp: Footprints, ops, sources: list[Source]) -> Tensor:
    return torch.stack([fp.column(ops.interior_source([s])) for s in unit(sources)], dim=1)


def main() -> None:
    import numpy as np

    from rectangle_report import heatmap, write_png

    coarse, fine = Region(), Region(nx=401, ny=201)
    reported = default_sources()
    pool = station_pool()
    stations = pool[:12]

    print("=== Обратная задача: источники загрязнения по станциям ===")
    print(f"Станций: {len(stations)}; «истина» на сетке {fine.nx}x{fine.ny}, "
          f"восстановление на {coarse.nx}x{coarse.ny}; шум {NOISE_REL:.0%} + {NOISE_ABS}, "
          f"ошибка модели в весах {MODEL_REL:.0%}")
    d, sigma = observe(fine, true_sources(), stations)

    fp_all = compute_footprints(coarse, pool)
    fp = Footprints(coarse, fp_all.phi[:12], fp_all.seconds, fp_all.iterations[:12])
    print(f"Следы станций: {len(pool)} сопряжённых задач, "
          f"{min(fp_all.iterations)}–{max(fp_all.iterations)} итераций, {fp_all.seconds:.2f} с")

    ops = build_operators(coarse)
    g = known_columns(fp, ops, reported)
    m = len(stations)

    # 1. Заявленные мощности согласуются с данными?
    pred = g @ torch.tensor([s.rate for s in reported], dtype=torch.float64)
    chi2_rep = torch.sum(((pred - d) / sigma) ** 2).item()
    print(f"\n1. Заявленные мощности: χ²/M = {chi2_rep / m:.1f} (≈1 если всё сходится)")

    # 2. Оценка мощностей только известных источников.
    fit = weighted_lsq(g, d, sigma)
    print(f"2. МНК по известным источникам: χ²/(M−K) = {fit.chi2 / (m - len(reported)):.1f}")
    for s, r, e in zip(reported, fit.rates, fit.std):
        print(f"   {s.name:9s}: {r.item():6.1f} ± {e.item():4.1f} (заявлено {s.rate:g}, "
              f"истина {TRUE_RATES[s.name]:g})")

    # 3. Поиск незаявленного источника: один набор шума подробно, затем 50 наборов.
    start = time.perf_counter()
    cand = fp.candidate_columns(HIDDEN.sigma)
    scan, one = locate(fp, g, d, sigma, cand)
    scan_s = time.perf_counter() - start
    print(f"3. Перебор {scan.misfit.numel()} мест-кандидатов: {scan_s * 1000:.0f} мс")
    print(f"   один набор шума: x = {scan.best_xy[0]:.1f}, y = {scan.best_xy[1]:.1f} км "
          f"(истина {HIDDEN.x}, {HIDDEN.y}; ошибка {one.error_km:.2f} км), "
          f"95 %-я область {one.area95:.1f} км², истина внутри: {'да' if one.truth_in_95 else 'нет'}")
    print(f"   χ²/(M−K−3) = {one.chi2 / (m - len(reported) - 3):.2f}")
    for name, r, e in zip([s.name for s in reported] + ["Новый источник"], one.rates, one.std):
        truth = TRUE_RATES.get(name, HIDDEN.rate)
        print(f"   {name:14s}: {r.item():6.1f} ± {e.item():4.1f} (истина {truth:g})")

    clean = clean_observations(fine, true_sources(), pool)
    runs = [locate(fp, g, *add_noise(clean[:m], seed), cand)[1] for seed in range(100, 150)]
    errs = np.array([r.error_km for r in runs])
    rates = np.array([r.rates.tolist() for r in runs])
    cover = np.mean([r.truth_in_95 for r in runs])
    q = lambda v: "{:.0f} [{:.0f}–{:.0f}]".format(np.median(v), *np.percentile(v, [25, 75]))
    print(f"   50 наборов шума: ошибка места медиана {np.median(errs):.2f} км, "
          f"< 1 км в {np.mean(errs < 1):.0%}, > 3 км в {np.mean(errs > 3):.0%}; "
          f"истина в 95 %-й области в {cover:.0%}")
    print(f"   мощности, медиана [квартили]: ТЭЦ {q(rates[:, 0])}, Промзона {q(rates[:, 1])}, "
          f"Автобан {q(rates[:, 2])}, пожар {q(rates[:, 3])}")

    # 4. Сколько станций нужно: случайные сети И случайный шум.
    print("\n4. Число станций -> медианы по 50 случайным сетям с разным шумом:")
    print("   станций | ошибка места, км | < 1 км | ошибка мощности пожара | ТЭЦ | истина в 95 %")
    g_all = known_columns(fp_all, ops, reported)
    cand_all = fp_all.candidate_columns(HIDDEN.sigma)
    gen = torch.Generator().manual_seed(3)
    for count in (4, 6, 8, 12, 16, 24):
        res = []
        for trial in range(50):
            idx = torch.randperm(len(pool), generator=gen)[:count]
            d_i, s_i = add_noise(clean[idx], 1000 * count + trial)
            sub = Footprints(coarse, fp_all.phi[idx], 0.0, [])
            res.append(locate(sub, g_all[idx], d_i, s_i, cand_all[idx])[1])
        e = np.array([r.error_km for r in res])
        rate_err = np.median([abs(r.rates[-1].item() - HIDDEN.rate) / HIDDEN.rate for r in res])
        plant = np.median([r.rates[0].item() for r in res])
        cov = np.mean([r.truth_in_95 for r in res])
        print(f"   {count:7d} | {np.median(e):16.2f} | {np.mean(e < 1):6.0%} | "
              f"{rate_err:22.0%} | {plant:3.0f} | {cov:13.0%}")

    # Картинки: правдоподобие места exp(−Δχ²/2), станции — белые, истина — красная.
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    like = torch.exp(-0.5 * (scan.misfit - scan.misfit.min()))
    img = heatmap(full_field(like).numpy())
    for x, y in stations:
        i, j = nearest_node(coarse, x, y)
        img[i:i + 3, j:j + 3] = 255
    i, j = nearest_node(coarse, HIDDEN.x, HIDDEN.y)
    img[i:i + 3, j:j + 3] = [230, 40, 40]
    write_png(out / "eco_inverse_location.png", img[::-1].copy(), 4)
    print("\nСохранено: results/eco_inverse_location.png "
          "(вероятность места пожара; белые — станции, красная — истинное место)")


if __name__ == "__main__":
    main()
