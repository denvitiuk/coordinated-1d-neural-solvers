import numpy as np
import torch


def laplace_residual(solution: np.ndarray) -> float:
    interior = solution[1:-1, 1:-1]

    residual = (
        solution[:-2, 1:-1]
        + solution[2:, 1:-1]
        + solution[1:-1, :-2]
        + solution[1:-1, 2:]
        - 4.0 * interior
    )

    return float(np.max(np.abs(residual)))


direct = np.load("direct_laplace_solution.npy")

u = torch.load(
    "coordinator_u_lbfgs.pt",
    map_location="cpu",
    weights_only=True,
).numpy()

v = torch.load(
    "coordinator_v_lbfgs.pt",
    map_location="cpu",
    weights_only=True,
).numpy()

# Берём среднее двух согласуемых внутренних решений.
coordinated = np.zeros_like(direct)
coordinated[1:-1, 1:-1] = 0.5 * (u + v)

x = np.linspace(0.0, 1.0, 100)
coordinated[-1, :] = np.sin(np.pi * x)

difference = coordinated - direct

print("=== Direct vs coordinator comparison ===")
print(f"Max |u_horizontal - u_vertical|: {np.max(np.abs(u - v)):.8e}")
print(f"RMS |u_horizontal - u_vertical|: {np.sqrt(np.mean((u - v) ** 2)):.8e}")
print()
print(f"Max |coordinated - direct|: {np.max(np.abs(difference)):.8e}")
print(f"RMS |coordinated - direct|: {np.sqrt(np.mean(difference ** 2)):.8e}")
print()
print(f"Direct Laplace residual: {laplace_residual(direct):.8e}")
print(f"Coordinator Laplace residual: {laplace_residual(coordinated):.8e}")