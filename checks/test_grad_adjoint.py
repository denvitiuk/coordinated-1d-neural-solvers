"""grad_adjoint против autograd, центральной разности и плотной формулы Mᵀ(Mq - c)."""

from common import DTYPE, build_general_problem, dense_reference, rel_err

import torch

from coordinator_laplace_descent import functional, grad_adjoint


def test_grad_adjoint() -> None:
    problem = build_general_problem(seed=0)
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(problem.fx.shape, generator=gen, dtype=DTYPE)

    g = grad_adjoint(problem, q)

    # 1) autograd как эталон (только здесь).
    q_ag = q.clone().requires_grad_(True)
    functional(problem.residual(q_ag)).backward()
    err_autograd = rel_err(g, q_ag.grad)

    # 2) центральная разность по каждой координате.
    eps = 1e-5
    g_fd = torch.zeros_like(q)
    with torch.no_grad():
        for idx in range(q.numel()):
            e = torch.zeros_like(q).reshape(-1)
            e[idx] = eps
            e = e.reshape(q.shape)
            j_plus = functional(problem.residual(q + e))
            j_minus = functional(problem.residual(q - e))
            g_fd.reshape(-1)[idx] = (j_plus - j_minus) / (2 * eps)
    err_fd = rel_err(g, g_fd)

    # 3) плотная формула, собранная без прогонки.
    m, c = dense_reference(problem)
    g_dense = (m.T @ (m @ q.reshape(-1) - c)).reshape(q.shape)
    err_dense = rel_err(g, g_dense)

    print(
        f"test_grad_adjoint: rel err vs autograd {err_autograd:.2e}, "
        f"vs central FD {err_fd:.2e}, vs dense Mᵀ(Mq-c) {err_dense:.2e}"
    )
    assert err_autograd < 1e-6
    assert err_fd < 1e-6
    assert err_dense < 1e-6
    print("test_grad_adjoint: OK")


if __name__ == "__main__":
    test_grad_adjoint()
