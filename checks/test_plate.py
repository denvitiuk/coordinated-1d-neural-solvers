"""
Пластина из диплома решена координацией:
  1. эталон exact_plate совпадает с переносом main.py диплома (16 x 36);
  2. координация на 16 x 36 совпадает с main.py;
  3. координация на 51 x 121 (сетка из текста) совпадает с точным решением,
     J монотонна, выбран GMRES + ADI-предобусловливание;
  4. результат не зависит от s, от s зависит только число итераций;
  5. без сдвига (s = 0) задача отклоняется — L_x вырождена.
"""

import common  # noqa: F401

import numpy as np
import torch

from plate_coordination import Plate, build_plate_problem, exact_plate, solve_plate
from thesis_main_direct import thesis_direct


def _monotone(history):
    return all(b <= a * (1 + 1e-12) for a, b in zip(history, history[1:]))


def test_plate() -> None:
    small = Plate(mx=16, ny=36)
    direct = thesis_direct(alx=5.0, aly=12.0, mx=16, ny=36, tu=60.0, td=20.0)

    ref_err = np.max(np.abs(exact_plate(small).numpy() - direct))
    t_small, res_small, _ = solve_plate(small)
    coord_err = np.max(np.abs(t_small.numpy() - direct))
    print(f"test_plate 16x36: exact vs thesis main.py {ref_err:.1e}, "
          f"coordination vs main.py {coord_err:.1e} ({res_small.iterations} it)")
    assert ref_err < 1e-9 and coord_err < 1e-8

    big = Plate()  # 51 x 121
    exact = exact_plate(big)
    t_big, res_big, problem = solve_plate(big)
    err = torch.max(torch.abs(t_big - exact)).item()
    print(f"test_plate 51x121: {res_big.method}, {res_big.iterations} it, "
          f"{res_big.tridiagonal_solves} solves, max|T - T*| = {err:.1e}")
    assert problem.m_is_symmetric() and res_big.method == "GMRES+P"
    assert res_big.converged and err < 1e-8 and _monotone(res_big.history)

    lam = big.lambda_min_y()
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        t_s, res_s, _ = solve_plate(big, shift=frac * lam)
        err_s = torch.max(torch.abs(t_s - exact)).item()
        print(f"  shift = {frac:.2f}·λmin: {res_s.iterations:>4} it, max|T - T*| = {err_s:.1e}")
        assert res_s.converged and err_s < 1e-7

    try:
        build_plate_problem(big, "cpu", torch.float64, shift=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("s = 0 must be rejected (singular Neumann matrix)")
    print("test_plate: OK")


if __name__ == "__main__":
    test_plate()
