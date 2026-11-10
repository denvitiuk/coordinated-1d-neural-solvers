"""CGLS на той же задаче: сходится к q* и за меньшее число итераций, чем спуск."""

from common import build_general_problem, exact_minimizer
from test_descent_exact_step import MAX_ITER, RTOL, run_descent

import torch

from coordinator_laplace_descent import cgls


def test_cgls() -> None:
    problem = build_general_problem(seed=0)
    q0 = torch.zeros_like(problem.fx)
    result = cgls(problem, q0, rtol=RTOL, max_iter=MAX_ITER)

    q_star = exact_minimizer(problem)
    error = torch.linalg.vector_norm(result.q - q_star).item()
    sd_iterations = run_descent()[1].iterations

    print(
        f"test_cgls: {result.iterations} iterations (SD: {sd_iterations}), "
        f"||q - q*|| = {error:.2e}"
    )
    assert result.converged, "CGLS did not reach rtol"
    assert error < 1e-6
    assert result.iterations < sd_iterations
    print("test_cgls: OK")


if __name__ == "__main__":
    test_cgls()
