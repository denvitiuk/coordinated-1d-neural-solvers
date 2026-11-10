import time

import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def solve_batched_tridiagonal(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """
    Решает batch независимых трёхдиагональных систем:

        lower[i] * x[i - 1]
        + diagonal[i] * x[i]
        + upper[i] * x[i + 1]
        = rhs[i]

    Размеры:
        lower, diagonal, upper, rhs: [batch_size, system_size]

    Алгоритм Томаса идёт последовательно по индексам системы,
    но все системы из batch обрабатываются параллельно.
    """

    batch_size, system_size = rhs.shape

    c_prime = torch.zeros_like(upper)
    d_prime = torch.zeros_like(rhs)

    c_prime[:, 0] = upper[:, 0] / diagonal[:, 0]
    d_prime[:, 0] = rhs[:, 0] / diagonal[:, 0]

    for i in range(1, system_size):
        denominator = diagonal[:, i] - lower[:, i] * c_prime[:, i - 1]

        if i < system_size - 1:
            c_prime[:, i] = upper[:, i] / denominator

        d_prime[:, i] = (
            rhs[:, i] - lower[:, i] * d_prime[:, i - 1]
        ) / denominator

    solution = torch.zeros_like(rhs)
    solution[:, -1] = d_prime[:, -1]

    for i in range(system_size - 2, -1, -1):
        solution[:, i] = d_prime[:, i] - c_prime[:, i] * solution[:, i + 1]

    return solution


DENSE_MAX_SIZE = 512


class TridiagonalFactor:
    """
    Прямой ход прогонки по матрице, выполненный один раз.

    Матрица в методе координатора не меняется, меняется только правая часть,
    поэтому c'[i] и 1/знаменатель[i] считаем заранее; solve() делает лишь
    проход по правой части и обратный ход (вдвое меньше операций).
    """

    def __init__(
        self,
        lower: torch.Tensor,
        diagonal: torch.Tensor,
        upper: torch.Tensor,
        method: str = "auto",
    ) -> None:
        """
        method: "auto" (dense при общей матрице и n <= DENSE_MAX_SIZE),
        "dense", "thomas" или "pcr" (параллельная циклическая редукция, для GPU).
        """
        system_size = diagonal.shape[1]
        self.method = method
        if method == "pcr":
            self.dense = False
            self._init_pcr(lower, diagonal, upper)
            return

        c_prime = torch.zeros_like(upper)
        inv_denominator = torch.zeros_like(diagonal)

        inv_denominator[:, 0] = 1.0 / diagonal[:, 0]
        c_prime[:, 0] = upper[:, 0] * inv_denominator[:, 0]

        for i in range(1, system_size):
            inv_denominator[:, i] = 1.0 / (
                diagonal[:, i] - lower[:, i] * c_prime[:, i - 1]
            )
            if i < system_size - 1:
                c_prime[:, i] = upper[:, i] * inv_denominator[:, i]

        self.lower = lower
        self.c_prime = c_prime
        self.inv_denominator = inv_denominator

        # Быстрый путь: A = L·U, L — нижняя двухдиагональная (знаменатели на
        # диагонали, lower под ней), U — единичная верхняя (c' над диагональю).
        # Решение = две треугольные системы одним вызовом BLAS вместо цикла
        # Python по n шагам. Плотные L, U занимают n² — выгодно при малых n.
        # В auto плотный путь берём только для одной общей матрицы на весь
        # пакет: при разных строках это batch·n² памяти и операций, и цикл
        # прогонки быстрее (замер на задаче загрязнения 401 x 201: 0.44 с
        # против 9.5 с).
        shared = all(
            bool(torch.equal(t, t[:1].expand_as(t)))
            for t in (lower, diagonal, upper)
        )
        self.dense = method == "dense" or (
            method == "auto" and shared and system_size <= DENSE_MAX_SIZE
        )
        if self.dense:
            rows = slice(0, 1) if shared else slice(None)
            idx = torch.arange(system_size, device=diagonal.device)

            l_mat = torch.diag_embed(1.0 / inv_denominator[rows])
            l_mat[:, idx[1:], idx[:-1]] = lower[rows, 1:]
            u_mat = torch.diag_embed(torch.ones_like(diagonal[rows]))
            u_mat[:, idx[:-1], idx[1:]] = c_prime[rows, :-1]

            self.shared = shared
            self.l_mat = l_mat[0] if shared else l_mat
            self.u_mat = u_mat[0] if shared else u_mat

    def _init_pcr(self, lower: torch.Tensor, diagonal: torch.Tensor, upper: torch.Tensor) -> None:
        """
        Параллельная циклическая редукция (Hockney, 1965; PCR).

        Шаг k (k = 1, 2, 4, ...): уравнение i исключает соседей i-k и i+k
        с помощью уравнений i-k и i+k:
            α_i = -a_i / b_{i-k},   γ_i = -c_i / b_{i+k},
            a_i ← α_i a_{i-k},  c_i ← γ_i c_{i+k},
            b_i ← b_i + α_i c_{i-k} + γ_i a_{i+k},
            d_i ← d_i + α_i d_{i-k} + γ_i d_{i+k}.
        После ⌈log₂ n⌉ шагов каждое уравнение — b_i x_i = d_i.
        Все i и все системы пакета обрабатываются одновременно — нет
        последовательного прохода по n, как в прогонке. Работа O(n log n)
        вместо O(n), но глубина O(log n) вместо O(n): на GPU это важнее.

        Матрица не меняется, поэтому α, γ и 1/b считаются один раз;
        solve() — только ⌈log₂ n⌉ обновлений правой части.
        Общая матрица на весь пакет хранится одной строкой [1, n].
        """
        shared = all(
            bool(torch.equal(t, t[:1].expand_as(t)))
            for t in (lower, diagonal, upper)
        )
        if shared:
            lower, diagonal, upper = lower[:1], diagonal[:1], upper[:1]
        a, b, c = lower.clone(), diagonal.clone(), upper.clone()
        n = b.shape[1]
        self.pcr_steps: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        k = 1
        while k < n:
            b_prev, b_next = _shift_down(b, k, 1.0), _shift_up(b, k, 1.0)
            alpha = -a / b_prev
            gamma = -c / b_next
            a_prev, c_prev = _shift_down(a, k, 0.0), _shift_down(c, k, 0.0)
            a_next, c_next = _shift_up(a, k, 0.0), _shift_up(c, k, 0.0)
            b = b + alpha * c_prev + gamma * a_next
            a = alpha * a_prev
            c = gamma * c_next
            self.pcr_steps.append((k, alpha, gamma))
            k *= 2
        self.pcr_inv_b = 1.0 / b

    def _solve_pcr(self, rhs: torch.Tensor) -> torch.Tensor:
        d = rhs
        for k, alpha, gamma in self.pcr_steps:
            d = d + alpha * _shift_down(d, k, 0.0) + gamma * _shift_up(d, k, 0.0)
        return d * self.pcr_inv_b

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        if self.method == "pcr":
            return self._solve_pcr(rhs)
        if self.dense:
            return self._solve_dense(rhs)
        return self._solve_thomas(rhs)

    def _solve_dense(self, rhs: torch.Tensor) -> torch.Tensor:
        if self.shared:
            # Одна матрица на весь batch: правые части — столбцы [n, batch].
            y = torch.linalg.solve_triangular(self.l_mat, rhs.T, upper=False)
            x = torch.linalg.solve_triangular(self.u_mat, y, upper=True, unitriangular=True)
            return x.T
        y = torch.linalg.solve_triangular(self.l_mat, rhs.unsqueeze(-1), upper=False)
        x = torch.linalg.solve_triangular(self.u_mat, y, upper=True, unitriangular=True)
        return x.squeeze(-1)

    def _solve_thomas(self, rhs: torch.Tensor) -> torch.Tensor:
        system_size = rhs.shape[1]
        lower, c_prime, inv_den = self.lower, self.c_prime, self.inv_denominator

        d_prime = torch.zeros_like(rhs)
        d_prime[:, 0] = rhs[:, 0] * inv_den[:, 0]
        for i in range(1, system_size):
            d_prime[:, i] = (rhs[:, i] - lower[:, i] * d_prime[:, i - 1]) * inv_den[:, i]

        solution = torch.zeros_like(rhs)
        solution[:, -1] = d_prime[:, -1]
        for i in range(system_size - 2, -1, -1):
            solution[:, i] = d_prime[:, i] - c_prime[:, i] * solution[:, i + 1]

        return solution


def _shift_down(t: torch.Tensor, k: int, fill: float) -> torch.Tensor:
    """out[:, i] = t[:, i - k], за краем — fill."""
    return torch.nn.functional.pad(t[:, :-k], (k, 0), value=fill)


def _shift_up(t: torch.Tensor, k: int, fill: float) -> torch.Tensor:
    """out[:, i] = t[:, i + k], за краем — fill."""
    return torch.nn.functional.pad(t[:, k:], (0, k), value=fill)


def transpose_tridiagonal(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Коэффициенты Aᵀ для каждой системы batch.

        A[i, i-1] = lower[i],  A[i, i+1] = upper[i]
        Aᵀ[i, i-1] = A[i-1, i] = upper[i-1]
        Aᵀ[i, i+1] = A[i+1, i] = lower[i+1]
    """

    lower_t = torch.zeros_like(lower)
    upper_t = torch.zeros_like(upper)

    lower_t[:, 1:] = upper[:, :-1]
    upper_t[:, :-1] = lower[:, 1:]

    return lower_t, diagonal.clone(), upper_t


def calculate_residual(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    solution: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """
    Возвращает max |A*x - rhs| для всего batch систем.
    """

    result = diagonal * solution

    result[:, 1:] += lower[:, 1:] * solution[:, :-1]
    result[:, :-1] += upper[:, :-1] * solution[:, 1:]

    return torch.max(torch.abs(result - rhs))


if __name__ == "__main__":
    device = get_device()
    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64

    # Для задачи Лапласа на сетке 100x100:
    # 98 внутренних строк / столбцов,
    # в каждой 98 внутренних узлов.
    batch_size = 98
    system_size = 98

    torch.manual_seed(42)

    lower = -torch.ones(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
    )

    diagonal = 4 * torch.ones(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
    )

    upper = -torch.ones(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
    )

    # На краях у системы нет соседей вне диапазона.
    lower[:, 0] = 0
    upper[:, -1] = 0

    # Создаём известное решение и строим правую часть.
    expected_solution = torch.rand(
        (batch_size, system_size),
        device=device,
        dtype=dtype,
    )

    rhs = diagonal * expected_solution
    rhs[:, 1:] += lower[:, 1:] * expected_solution[:, :-1]
    rhs[:, :-1] += upper[:, :-1] * expected_solution[:, 1:]

    synchronize(device)
    start = time.perf_counter()

    solution = solve_batched_tridiagonal(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        rhs=rhs,
    )

    synchronize(device)
    elapsed = time.perf_counter() - start

    residual = calculate_residual(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        solution=solution,
        rhs=rhs,
    )

    max_solution_error = torch.max(
        torch.abs(solution - expected_solution)
    ).item()

    print("=== Batched tridiagonal solver ===")
    print(f"Device: {device}")
    print(f"Systems in batch: {batch_size}")
    print(f"System size: {system_size}")
    print(f"Time: {elapsed:.6f} sec")
    print(f"Max residual |A*x - b|: {residual.item():.8e}")
    print(f"Max solution error: {max_solution_error:.8e}")