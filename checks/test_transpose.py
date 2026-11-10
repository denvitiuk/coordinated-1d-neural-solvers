"""transpose_tridiagonal даёт ровно плотную Aᵀ, и прогонка с ней решает Aᵀx = b."""

from common import DTYPE, dense_tridiagonal

import torch

from batched_tridiagonal import solve_batched_tridiagonal, transpose_tridiagonal


def test_transpose() -> None:
    gen = torch.Generator().manual_seed(1)
    batch, size = 3, 6

    lo = torch.randn((batch, size), generator=gen, dtype=DTYPE)
    up = torch.randn((batch, size), generator=gen, dtype=DTYPE)
    lo[:, 0] = 0
    up[:, -1] = 0
    di = 5.0 + torch.rand((batch, size), generator=gen, dtype=DTYPE)

    lo_t, di_t, up_t = transpose_tridiagonal(lo, di, up)
    rhs = torch.randn((batch, size), generator=gen, dtype=DTYPE)
    x = solve_batched_tridiagonal(lo_t, di_t, up_t, rhs)

    for k in range(batch):
        a = dense_tridiagonal(lo[k], di[k], up[k])
        a_t = dense_tridiagonal(lo_t[k], di_t[k], up_t[k])

        assert torch.equal(a_t, a.T), f"batch {k}: transposed coefficients != dense .T"
        assert lo_t[k, 0] == 0 and up_t[k, -1] == 0

        expected = torch.linalg.solve(a.T, rhs[k])
        err = torch.max(torch.abs(x[k] - expected)).item()
        assert err < 1e-12, f"batch {k}: solve with Aᵀ error {err:.2e}"

    print("test_transpose: OK")


if __name__ == "__main__":
    test_transpose()
