"""
Перенос main.py из диплома (прямое решение для пластины) на numpy, без matplotlib/scipy.

Сборка матрицы AA и правой части BR — построчно как в дипломе:
  нумерация неизвестных k = j + i·(NY-2), i — индекс по x, j — по y СВЕРХУ вниз;
  ∂T/∂n = 0 слева и справа через T_0 = T_1 (диагональ 2·DX² + DY² в крайних блоках);
  сверху T = TU·cos(πx/ALX), снизу T = TD·sin(πx/ALX).
Возвращает ZZ [NY, MX] как в дипломе: строка 0 — низ (y = 0), строка NY-1 — верх.
"""

from math import cos, pi, sin

import numpy as np


def fii(alx, tu, x):
    return tu * cos(x * pi / alx)


def psii(alx, td, x):
    return td * sin(x * pi / alx)


def thesis_direct(alx=5.0, aly=12.0, mx=16, ny=36, tu=60.0, td=20.0) -> np.ndarray:
    dx = alx / (mx - 1)
    dy = aly / (ny - 1)
    nm = (mx - 2) * (ny - 2)
    aa = np.zeros((nm, nm))
    br = np.zeros(nm)
    dxdx, dydy = dx * dx, dy * dy

    for i in range(0, ny - 2):
        aa[i, i] = 2.0 * dxdx + dydy
    for i in range(ny - 2, (ny - 2) * (mx - 3)):
        aa[i, i] = 2.0 * (dxdx + dydy)
    for i in range((mx - 3) * (ny - 2), (mx - 2) * (ny - 2)):
        aa[i, i] = 2.0 * dxdx + dydy
    for i in range(0, (mx - 3) * (ny - 2)):
        j = i + ny - 2
        aa[i, j] = aa[j, i] = -dydy
    for i in range(0, nm - 1):
        aa[i, i + 1] = aa[i + 1, i] = -dxdx
    for i in range(1, mx - 2):
        j = (ny - 2) * i - 1
        aa[j, j + 1] = aa[j + 1, j] = 0.0

    for i in range(0, mx - 2):
        j = (ny - 2) * i
        x = dx * (i + 1)
        br[j] = fii(alx, tu, x) * dxdx
        br[j + ny - 3] = psii(alx, td, x) * dxdx

    xx = np.linalg.solve(aa, br)  # в дипломе — Холецкий; результат тот же

    zz = np.zeros((ny, mx))
    for i in range(mx):
        zz[ny - 1, i] = fii(alx, tu, i * dx)
        zz[0, i] = psii(alx, td, i * dx)
    for j in range(2, ny):
        for i in range(1, mx - 1):
            zz[ny - j, i] = xx[j - 2 + (i - 1) * (ny - 2)]
        zz[ny - j, 0] = zz[ny - j, 1]
        zz[ny - j, mx - 1] = zz[ny - j, mx - 2]
    return zz


if __name__ == "__main__":
    np.set_printoptions(linewidth=140)
    for ny in (36, 37):
        zz = thesis_direct(ny=ny)
        print(f"NY = {ny}: first interior row from top:\n{np.round(zz[-2], 2)}")
    print("thesis printout:\n[ 46.6 46.6 43.94 39.13 32.5 24.42 15.25 5.42 -4.64 -14.47 ...]")
