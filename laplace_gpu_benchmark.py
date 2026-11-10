"""
Задание: оценить эффективность метода координации на сетке 100 x 100
сравнением затрат при традиционном решении двумерной задачи и предлагаемым
методом с GPU, при конечно-разностных схемах одного порядка.

Задача: Δφ = 0 в квадрате [0, 1]², φ = 1 сверху, 0 на остальных сторонах
(как в прежнем сравнении). φ = sin(πx) не годится: это одна собственная функция
по x, и координация без предобусловливания сходится за ~8 итераций — нечестно.
Все методы решают ОДНУ И ТУ ЖЕ пятиточечную схему (второй порядок), поэтому
ошибка считается относительно точного решения этой разностной схемы
(дискретный синус-ряд, все N-2 гармоник):
    φ_ij = Σ_k ĝ_k sin(kπ x_j) · sinh(μ_k i) / sinh(μ_k (N-1)),
    cosh μ_k = 1 + 2 sin²(kπh / 2),   ĝ_k = 2/(N-1) Σ_j g_j sin(kπ x_j).

Традиционные методы (двумерная схема целиком):
  Якоби           — простейший итерационный, CPU;
  CG (5 точек)    — сопряжённые градиенты без матрицы, CPU float64 и GPU float32;
  Холецкий        — прямое решение плотной матрицы, CPU (только малые сетки).
Метод координации (только одномерные прогонки):
  CR              — без предобусловливания (близко к диплому);
  GMRES + ADI     — предобусловленный вариант;
  прогонка: thomas (цикл), dense (треугольные BLAS), pcr (циклическая редукция).

Каждый вариант считается в отдельном процессе: время (лучшее из 3 после
прогрева), итерации, ошибка, прирост памяти процесса (RSS), память GPU.
Пишет results/laplace_gpu_benchmark.csv и .md.

    python laplace_gpu_benchmark.py            # все сетки
    python laplace_gpu_benchmark.py 100        # только 100 x 100
    python laplace_gpu_benchmark.py 1600       # большая сетка (имеет смысл на NVIDIA)

На NVIDIA (CUDA) добавляются GPU-варианты во float64 — сравнение на равной
точности с CPU. Как запустить — RUN_NVIDIA.md.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from batched_tridiagonal import get_device, synchronize
from coordinator_laplace_descent import (
    AdiPreconditioner,
    assemble_solution,
    build_laplace_problem,
    conjugate_residual,
    gmres,
)

GRIDS = [100, 200, 400, 800]
REPEATS = 3
GPU = get_device()

# имя -> (семейство, устройство, точность, наибольшая сетка)
VARIANTS = {
    "Якоби": ("jacobi", "cpu", "f64", 100),
    "CG 5 точек": ("cg", "cpu", "f64", 800),
    "CG 5 точек GPU": ("cg", GPU, "f32", 800),
    "Холецкий (плотная)": ("cholesky", "cpu", "f64", 100),
    "Коорд. CR, прогонка": ("cr:thomas", "cpu", "f64", 200),
    "Коорд. GMRES+ADI, BLAS": ("adi:auto", "cpu", "f64", 800),
    "Коорд. GMRES+ADI, PCR": ("adi:pcr", "cpu", "f64", 800),
    "Коорд. CR, PCR GPU": ("cr:pcr", GPU, "f32", 400),
    "Коорд. GMRES+ADI, BLAS GPU": ("adi:dense", GPU, "f32", 800),
    "Коорд. GMRES+ADI, PCR GPU": ("adi:pcr", GPU, "f32", 800),
}
if GPU == "cuda":
    # NVIDIA умеет float64 — сравнение с CPU на равной точности.
    VARIANTS.update({
        "CG 5 точек GPU f64": ("cg", GPU, "f64", 1600),
        "Коорд. GMRES+ADI, BLAS GPU f64": ("adi:dense", GPU, "f64", 1600),
        "Коорд. GMRES+ADI, PCR GPU f64": ("adi:pcr", GPU, "f64", 1600),
    })
    for _name in ("CG 5 точек", "Коорд. GMRES+ADI, BLAS", "Коорд. GMRES+ADI, PCR",
                  "CG 5 точек GPU", "Коорд. GMRES+ADI, PCR GPU"):
        _f, _d, _p, _ = VARIANTS[_name]
        VARIANTS[_name] = (_f, _d, _p, 1600)


def peak_rss_bytes() -> int:
    """Пиковая память процесса: macOS — байты, Linux — КиБ, Windows — через WinAPI."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb)
        return int(c.PeakWorkingSetSize)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * 1024 if platform.system() == "Linux" else peak


def top_values(x: torch.Tensor, lx: float = 1.0) -> torch.Tensor:
    return torch.ones_like(x)


def exact_discrete(n: int) -> torch.Tensor:
    """Точное решение пятиточечной схемы (hx = hy = h) дискретным синус-рядом."""
    h = 1.0 / (n - 1)
    j = torch.arange(1, n - 1, dtype=torch.float64)  # внутренние узлы по x
    k = torch.arange(1, n - 1, dtype=torch.float64)  # гармоники
    s = torch.sin(torch.pi * k[:, None] * j[None, :] / (n - 1))  # [K, J]
    g = top_values(j * h)
    g_hat = 2.0 / (n - 1) * (s @ g)
    mu = torch.acosh(1.0 + 2.0 * torch.sin(torch.pi * k * h / 2) ** 2)
    i = torch.arange(n, dtype=torch.float64)
    # sinh(μ i)/sinh(μ (N-1)) без переполнения.
    e = (i[:, None] - (n - 1)) * mu[None, :]
    ratio = torch.exp(e) * (-torch.expm1(-2 * mu[None, :] * i[:, None])) / (
        -torch.expm1(-2 * mu[None, :] * (n - 1)))
    full = torch.zeros((n, n), dtype=torch.float64)
    full[:, 1:-1] = (ratio * g_hat[None, :]) @ s
    return full


# --- традиционные методы: двумерная пятиточечная схема ----------------------


def _laplace5(phi_in: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
    """−Δ_h φ · h² для внутренних узлов (граница: сверху top, иначе 0)."""
    p = torch.nn.functional.pad(phi_in, (1, 1, 1, 1))
    p[-1, 1:-1] = top
    return 4 * phi_in - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


def solve_jacobi(n, device, dtype, rtol):
    top = top_values(torch.linspace(0, 1, n, device=device, dtype=dtype))[1:-1]
    phi = torch.zeros((n - 2, n - 2), device=device, dtype=dtype)
    zero = torch.zeros_like(phi)
    r0 = torch.linalg.vector_norm(_laplace5(zero, top))
    for it in range(1, 2_000_000):
        r = _laplace5(phi, top)
        phi = phi - r / 4
        if it % 50 == 0 and torch.linalg.vector_norm(_laplace5(phi, top)) <= rtol * r0:
            return phi, it
    return phi, it


def solve_cg(n, device, dtype, rtol):
    """CG для A φ = b, A = −Δ_h·h² (СПД), без хранения матрицы."""
    top = top_values(torch.linspace(0, 1, n, device=device, dtype=dtype))[1:-1]
    zero_top = torch.zeros_like(top)
    a = lambda v: _laplace5(v, zero_top)
    b = -_laplace5(torch.zeros((n - 2, n - 2), device=device, dtype=dtype), top)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = torch.sum(r * r)
    stop = (rtol * torch.linalg.vector_norm(b)) ** 2
    for it in range(1, 100_000):
        ap = a(p)
        alpha = rr / torch.sum(p * ap)
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = torch.sum(r * r)
        if it % 10 == 0 and rr_new.item() <= stop.item():
            return x, it
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, it


def solve_cholesky(n, device, dtype, rtol):
    m = n - 2
    size = m * m
    eye = torch.eye(m, dtype=dtype)
    t = 2 * eye - torch.diag(torch.ones(m - 1, dtype=dtype), 1) - torch.diag(torch.ones(m - 1, dtype=dtype), -1)
    a = torch.kron(eye, t) + torch.kron(t, eye)
    top = top_values(torch.linspace(0, 1, n, dtype=dtype))[1:-1]
    b = torch.zeros((m, m), dtype=dtype)
    b[-1] = top
    chol = torch.linalg.cholesky(a)
    x = torch.cholesky_solve(b.reshape(size, 1), chol).reshape(m, m)
    return x, 1


# --- метод координации ------------------------------------------------------


def solve_coordination(family, n, device, dtype, rtol):
    method, solver = family.split(":")
    pr = build_laplace_problem(n, n, device, dtype, top=top_values)
    pr.solver = solver
    pr.__post_init__()
    q0 = torch.zeros_like(pr.fx)
    if method == "adi":
        res = gmres(pr, q0, rtol=rtol, max_iter=500, preconditioner=AdiPreconditioner(pr))
    else:
        res = conjugate_residual(pr, q0, rtol=rtol, max_iter=50_000)
    full = assemble_solution(pr, res.q, top=top_values)
    return full[1:-1, 1:-1], res.iterations, pr.tridiagonal_solves


def run_one(name: str, n: int) -> dict:
    family, device, prec, _ = VARIANTS[name]
    dtype = torch.float64 if prec == "f64" else torch.float32
    rtol = 1e-10 if prec == "f64" else 1e-5

    def once():
        solves = None
        if family == "jacobi":
            phi, it = solve_jacobi(n, device, dtype, rtol)
        elif family == "cg":
            phi, it = solve_cg(n, device, dtype, rtol)
        elif family == "cholesky":
            phi, it = solve_cholesky(n, device, dtype, rtol)
        else:
            phi, it, solves = solve_coordination(family, n, device, dtype, rtol)
        return phi, it, solves

    rss0 = peak_rss_bytes()
    once()  # прогрев: компиляция ядер GPU, кэши
    best = float("inf")
    for _ in range(REPEATS if n <= 400 else 1):
        synchronize(device)
        start = time.perf_counter()
        phi, it, solves = once()
        synchronize(device)
        best = min(best, time.perf_counter() - start)
    rss1 = peak_rss_bytes()
    gpu_mib = None
    if device == "mps":
        gpu_mib = torch.mps.driver_allocated_memory() / 2**20
    elif device == "cuda":
        gpu_mib = torch.cuda.max_memory_allocated() / 2**20
    err = (phi.cpu().double() - exact_discrete(n)[1:-1, 1:-1]).abs().max().item()
    return {
        "grid": f"{n}x{n}", "unknowns": (n - 2) ** 2, "method": name, "device": device,
        "precision": prec, "seconds": best, "iterations": it, "tridiagonal_solves": solves,
        "max_error": err, "ram_mib": (rss1 - rss0) / 2**20, "gpu_mib": gpu_mib,
    }


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        print(json.dumps(run_one(sys.argv[2], int(sys.argv[3]))))
        return
    grids = [int(a) for a in sys.argv[1:]] or GRIDS
    gpu_name = {"cuda": lambda: torch.cuda.get_device_name(0), "mps": lambda: "Apple MPS"}.get(GPU, lambda: "нет")()
    print(f"PyTorch {torch.__version__}, CPU: {platform.processor() or platform.machine()}, "
          f"GPU: {gpu_name} ({GPU})", flush=True)
    rows = []
    for n in grids:
        for name, (_, device, _, max_n) in VARIANTS.items():
            if n > max_n or device == "cpu" and name.endswith("GPU"):
                continue
            out = subprocess.run([sys.executable, __file__, "--one", name, str(n)],
                                 capture_output=True, text=True, timeout=1800)
            if out.returncode != 0:
                print(f"{n} {name}: ОШИБКА\n{out.stderr[-2000:]}", file=sys.stderr)
                continue
            row = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(f"{row['grid']:>8} | {name:28s} | {row['seconds']:8.4f} с | "
                  f"{row['iterations']:>6} it | err {row['max_error']:.1e} | "
                  f"RAM +{row['ram_mib']:.0f} МиБ | GPU {row['gpu_mib'] or 0:.0f} МиБ", flush=True)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "laplace_gpu_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Сохранено: results/laplace_gpu_benchmark.csv")


if __name__ == "__main__":
    main()
