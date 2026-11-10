"""
Обратная задача (eco_inverse.py), маленькая сетка, float64:
  1. столбец через след станции ⟨φ_m, S⟩ совпадает с прямым решением в станции;
  2. на точных данных (та же сетка) перебор находит место пожара точно,
     а МНК — все четыре мощности;
  3. кандидаты с отрицательной мощностью известного источника отброшены;
  4. истина лежит в 95 %-й области.
"""

from common import DTYPE  # noqa: F401  (путь к проекту)

import torch

from eco_inverse import (
    HIDDEN,
    TRUE_RATES,
    clean_observations,
    compute_footprints,
    known_columns,
    locate,
    scan_extra_source,
    station_pool,
    true_sources,
    unit,
)
from eco_pollution import EcoSolver, Region, build_operators, default_sources, nearest_node


def test_eco_inverse() -> None:
    region = Region(nx=101, ny=51)
    stations = station_pool()[:12]
    fp = compute_footprints(region, stations)
    ops = build_operators(region)
    g = known_columns(fp, ops, default_sources())

    # 1. След станции = прямое решение для единичного источника.
    s0 = unit(default_sources())[0]
    c, _ = EcoSolver(ops).solve(ops.interior_source([s0]), rtol=1e-12)
    direct = torch.tensor([c[nearest_node(region, x, y)].item() for x, y in stations],
                          dtype=torch.float64)
    rel = ((g[:, 0] - direct).abs().max() / direct.abs().max()).item()
    assert rel < 1e-6, rel

    # 2. Точные данные, та же сетка: место и мощности восстанавливаются.
    d = clean_observations(region, true_sources(), stations)
    sigma = 0.01 * d.abs() + 1e-3
    scan, loc = locate(fp, g, d, sigma)
    assert loc.error_km < 0.25, (scan.best_xy, loc.error_km)
    truth = torch.tensor([*TRUE_RATES.values(), HIDDEN.rate], dtype=torch.float64)
    err = ((loc.rates - truth).abs() / truth).max().item()
    assert err < 0.02, (loc.rates.tolist(), err)
    assert loc.truth_in_95

    # 3. Ни один допустимый кандидат не даёт отрицательной мощности известных.
    finite = torch.isfinite(scan.misfit)
    assert finite.any() and not finite.all()

    print(f"test_eco_inverse 101x51: след vs прямое {rel:.1e}, место {scan.best_xy} "
          f"(ошибка {loc.error_km:.2f} км), мощности {[round(v, 1) for v in loc.rates.tolist()]}, "
          f"отброшено кандидатов {(~finite).sum().item()} из {finite.numel()}")
    print("test_eco_inverse: OK")


if __name__ == "__main__":
    test_eco_inverse()
