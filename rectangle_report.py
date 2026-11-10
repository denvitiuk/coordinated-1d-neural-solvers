"""
Лаплас на прямоугольниках: сравнение SD / CGLS / CR / GMRES (float64, CPU).
Сохраняет в results/:
  - rectangle_laplace.md / .csv                  сводная таблица методов
  - rectangle_<top>_<nx>x<ny>_solution.npy       решение U [ny, nx] (метод coordinate)
  - rectangle_<top>_<nx>x<ny>.png                тепловая карта U (y = 0 внизу)
"""

from __future__ import annotations

import csv
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "checks"))

from coordinator_laplace_descent import (  # noqa: E402
    assemble_solution,
    build_laplace_problem,
    coordinate,
)
from run_laplace import CASES, MAX_ITER, TOPS, run_laplace  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RTOL = 1e-10
DTYPE = torch.float64


def write_png(path: Path, image: np.ndarray, scale: int) -> None:
    """Пишет RGB PNG без matplotlib. image: [h, w, 3] uint8."""
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def heatmap(values: np.ndarray) -> np.ndarray:
    """0 → тёмно-синий, 1 → жёлтый (линейная палитра по 3 опорным цветам)."""
    t = np.clip((values - values.min()) / (np.ptp(values) or 1.0), 0.0, 1.0)
    stops = np.array([[20, 30, 90], [30, 150, 140], [250, 230, 60]], dtype=float)
    pos = t * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    return (stops[lo] * (1 - frac) + stops[lo + 1] * frac).astype(np.uint8)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for old in RESULTS.glob("rectangle_[0-9]*"):
        old.unlink()  # файлы прошлой версии отчёта (только sin)

    table = run_laplace()

    for top_name, top in TOPS.items():
        for nx, ny, lx, ly in CASES:
            problem = build_laplace_problem(nx, ny, "cpu", DTYPE, lx=lx, ly=ly, top=top)
            res = coordinate(problem, torch.zeros_like(problem.fx), rtol=RTOL)
            solution = assemble_solution(problem, res.q, lx, top).numpy()
            name = f"rectangle_{top_name}_{nx}x{ny}"
            np.save(RESULTS / f"{name}_solution.npy", solution)
            write_png(RESULTS / f"{name}.png", heatmap(solution[::-1]),
                      max(1, 400 // max(nx, ny)))

    with open(RESULTS / "rectangle_laplace.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    lines = [
        "# Laplace on rectangles: coordinating two families of 1D systems",
        "",
        f"Top boundary: `sin` = sin(πx/Lx) (one harmonic), `const` = 1 (all harmonics). "
        f"float64, CPU, rtol = {RTOL:g}, max_iter = {MAX_ITER}.",
        "Solves = batched tridiagonal solves (one pass over all rows or all columns). "
        "Error = max |U − U*| vs the exact solution of the same 5-point scheme.",
        "",
        "| Top | Grid (x·y) | Domain | Method | Iterations | Solves | Time, s | Error | Converged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['top']} | {r['grid']} | {r['domain']} | {r['method']} | {r['iterations']} "
            f"| {r['tridiagonal_solves']} | {r['seconds']:.3f} | {r['max_err']:.1e} "
            f"| {'yes' if r['converged'] else 'no'} |"
        )
    (RESULTS / "rectangle_laplace.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
