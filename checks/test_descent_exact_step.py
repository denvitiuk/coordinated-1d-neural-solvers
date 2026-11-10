"""Спуск с точным шагом: J монотонно убывает и сходится к q* из плотного lstsq."""

from common import build_general_problem, exact_minimizer

import torch

from coordinator_laplace_descent import steepest_descent

RTOL = 1e-12
MAX_ITER = 50_000


def run_descent():
    problem = build_general_problem(seed=0)
    q0 = torch.zeros_like(problem.fx)
    return problem, steepest_descent(problem, q0, rtol=RTOL, max_iter=MAX_ITER)


def test_descent_exact_step() -> int:
    problem, result = run_descent()
    q_star = exact_minimizer(problem)
    error = torch.linalg.vector_norm(result.q - q_star).item()

    history = result.history
    increases = sum(1 for a, b in zip(history, history[1:]) if b > a * (1 + 1e-12))

    print(
        f"test_descent_exact_step: {result.iterations} iterations, "
        f"||g||/||g0|| = {result.relative_measure:.2e}, ||q - q*|| = {error:.2e}, "
        f"J: {history[0]:.4e} -> {history[-1]:.4e}"
    )
    assert result.converged, "descent did not reach rtol"
    assert increases == 0, f"J increased {increases} times"
    assert error < 1e-6
    print("test_descent_exact_step: OK")
    return result.iterations


if __name__ == "__main__":
    test_descent_exact_step()
