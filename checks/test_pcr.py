"""
Параллельная циклическая редукция (TridiagonalFactor method="pcr"):
совпадает с прогонкой для несимметричных систем, общей и разных матриц,
n не степень двойки и n = 1; на GPU (если есть) — во float32.
Ещё: метод координации с PCR даёт то же решение Лапласа, что и с прогонкой.
"""

from common import DTYPE

import torch

from batched_tridiagonal import TridiagonalFactor, get_device
from coordinator_laplace_descent import assemble_solution, build_laplace_problem, coordinate


def _system(batch: int, n: int, shared: bool, dtype, device):
    gen = torch.Generator().manual_seed(n * 7 + batch)
    rows = 1 if shared else batch
    lo = -torch.rand(rows, n, generator=gen, dtype=torch.float64) - 0.2
    up = -torch.rand(rows, n, generator=gen, dtype=torch.float64) * 1.5
    di = lo.abs() + up.abs() + 0.1 + torch.rand(rows, n, generator=gen, dtype=torch.float64)
    lo[:, 0] = 0
    up[:, -1] = 0
    if shared:
        lo, di, up = (t.expand(batch, n).clone() for t in (lo, di, up))
    rhs = torch.randn(batch, n, generator=gen, dtype=torch.float64)
    return [t.to(device=device, dtype=dtype) for t in (lo, di, up, rhs)]


def test_pcr() -> None:
    worst = 0.0
    for n in (1, 2, 7, 64, 100, 257):
        for shared in (True, False):
            lo, di, up, rhs = _system(5, n, shared, DTYPE, "cpu")
            ref = TridiagonalFactor(lo, di, up, method="thomas").solve(rhs)
            got = TridiagonalFactor(lo, di, up, method="pcr").solve(rhs)
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, err)
            assert err < 1e-12, (n, shared, err)

    device = get_device()
    gpu_err = None
    if device != "cpu":
        lo, di, up, rhs = _system(98, 98, True, torch.float32, device)
        ref = TridiagonalFactor(*(t.cpu().double() for t in (lo, di, up)), method="thomas")
        ref = ref.solve(rhs.cpu().double())
        got = TridiagonalFactor(lo, di, up, method="pcr").solve(rhs).cpu().double()
        gpu_err = ((got - ref).abs().max() / ref.abs().max()).item()
        assert gpu_err < 1e-5, gpu_err

    fields = []
    for solver in ("thomas", "pcr"):
        pr = build_laplace_problem(40, 30, "cpu", DTYPE)
        pr.solver = solver
        pr.__post_init__()
        res = coordinate(pr, torch.zeros_like(pr.fx), rtol=1e-12)
        assert res.converged
        fields.append(assemble_solution(pr, res.q))
    lap = (fields[0] - fields[1]).abs().max().item()
    assert lap < 1e-10, lap

    print(f"test_pcr: max rel err vs Thomas {worst:.1e} (CPU float64), "
          f"GPU float32 {gpu_err if gpu_err is None else f'{gpu_err:.1e}'}, "
          f"Laplace via coordination PCR vs Thomas {lap:.1e}")
    print("test_pcr: OK")


if __name__ == "__main__":
    test_pcr()
