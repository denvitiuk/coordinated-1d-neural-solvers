

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


TIME_PATTERN = re.compile(r"^Time:\s*([0-9.]+)\s*sec", re.MULTILINE)
ITERATIONS_PATTERN = re.compile(r"^Iterations:\s*(\d+)", re.MULTILINE)
CG_ITERATIONS_PATTERN = re.compile(r"^CG iterations:\s*(\d+)", re.MULTILINE)
LAPLACE_RESIDUAL_PATTERN = re.compile(
    r"^Laplace residual:\s*([0-9.eE+-]+)",
    re.MULTILINE,
)
FINAL_ERROR_PATTERN = re.compile(r"^Final error:\s*([0-9.eE+-]+)", re.MULTILINE)
MAX_DIFFERENCE_PATTERN = re.compile(
    r"^Max \|coordinator - direct\|:\s*([0-9.eE+-]+)",
    re.MULTILINE,
)
RMS_DIFFERENCE_PATTERN = re.compile(
    r"^RMS \|coordinator - direct\|:\s*([0-9.eE+-]+)",
    re.MULTILINE,
)
MAX_RSS_PATTERN = re.compile(
    r"(\d+)\s+maximum resident set size",
    re.IGNORECASE,
)


def require_match(pattern: re.Pattern[str], text: str, field_name: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"Could not parse '{field_name}' from program output.\n\n{text}")

    return match.group(1)


def run_with_memory_measurement(script_name: str) -> tuple[str, int | None]:
    """
    Runs one benchmark in a separate process.

    On macOS, /usr/bin/time -l reports the peak resident-set size.
    This measures process RAM; it does not represent separate NVIDIA VRAM.
    """

    command = [
        "/usr/bin/time",
        "-l",
        sys.executable,
        script_name,
    ]

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    combined_output = f"{process.stdout}\n{process.stderr}"

    if process.returncode != 0:
        raise RuntimeError(
            f"Benchmark '{script_name}' failed with exit code {process.returncode}.\n\n"
            f"{combined_output}"
        )

    rss_match = MAX_RSS_PATTERN.search(combined_output)
    peak_rss_bytes = int(rss_match.group(1)) if rss_match else None

    return process.stdout, peak_rss_bytes


def bytes_to_megabytes(value: int | None) -> float | None:
    if value is None:
        return None

    return value / (1024 * 1024)


def write_csv(rows: list[dict[str, str]]) -> Path:
    output_path = OUTPUT_DIR / "benchmark_100x100.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_markdown_report(
    direct_output: str,
    coordinator_output: str,
    rows: list[dict[str, str]],
    max_difference: str,
    rms_difference: str,
) -> Path:
    output_path = OUTPUT_DIR / "benchmark_100x100.md"

    direct_row, coordinator_row = rows

    markdown = f"""# Benchmark: equation of Laplace on a 100 × 100 grid

## Methods

1. **Direct CPU method**: Jacobi iteration with the 5-point finite-difference stencil.
2. **Coordinator method**: 98 horizontal and 98 vertical tridiagonal systems, solved in batches; coordination uses conjugate gradients.

## Results

| Metric | Direct CPU Jacobi | Coordinator MPS |
|---|---:|---:|
| Time, sec | {direct_row['time_sec']} | {coordinator_row['time_sec']} |
| Iterations | {direct_row['iterations']} | {coordinator_row['iterations']} |
| Laplace residual | {direct_row['laplace_residual']} | {coordinator_row['laplace_residual']} |
| Peak process memory, MiB | {direct_row['peak_process_memory_mib']} | {coordinator_row['peak_process_memory_mib']} |

## Agreement of final solutions

- Max |coordinator − direct|: `{max_difference}`
- RMS |coordinator − direct|: `{rms_difference}`

## Notes

- Peak process memory is measured by macOS `/usr/bin/time -l` and reflects process RAM. It is not a direct measurement of separate NVIDIA VRAM.
- For this 100 × 100 task, MPS overhead may dominate runtime. The coordinator method's parallel structure is the batched solution of 98 horizontal and 98 vertical independent tridiagonal systems at each conjugate-gradient iteration.

## Raw output: direct CPU solver

```text
{direct_output.strip()}
```

## Raw output: coordinator solver

```text
{coordinator_output.strip()}
```
"""

    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def main() -> None:
    print("Running direct CPU benchmark...")
    direct_output, direct_peak_rss = run_with_memory_measurement(
        "direct_laplace_cpu.py"
    )

    print("Running coordinator MPS benchmark...")
    coordinator_output, coordinator_peak_rss = run_with_memory_measurement(
        "coordinator_laplace_cg.py"
    )

    direct_time = require_match(TIME_PATTERN, direct_output, "direct time")
    direct_iterations = require_match(
        ITERATIONS_PATTERN,
        direct_output,
        "direct iterations",
    )
    direct_error = require_match(
        FINAL_ERROR_PATTERN,
        direct_output,
        "direct final error",
    )

    coordinator_time = require_match(
        TIME_PATTERN,
        coordinator_output,
        "coordinator time",
    )
    coordinator_iterations = require_match(
        CG_ITERATIONS_PATTERN,
        coordinator_output,
        "coordinator CG iterations",
    )
    coordinator_residual = require_match(
        LAPLACE_RESIDUAL_PATTERN,
        coordinator_output,
        "coordinator Laplace residual",
    )
    max_difference = require_match(
        MAX_DIFFERENCE_PATTERN,
        coordinator_output,
        "maximum solution difference",
    )
    rms_difference = require_match(
        RMS_DIFFERENCE_PATTERN,
        coordinator_output,
        "RMS solution difference",
    )

    direct_memory_mib = bytes_to_megabytes(direct_peak_rss)
    coordinator_memory_mib = bytes_to_megabytes(coordinator_peak_rss)

    rows = [
        {
            "method": "Direct CPU Jacobi",
            "device": "cpu",
            "grid": "100x100",
            "time_sec": direct_time,
            "iterations": direct_iterations,
            "laplace_residual": direct_error,
            "peak_process_memory_mib": (
                f"{direct_memory_mib:.2f}" if direct_memory_mib is not None else "n/a"
            ),
        },
        {
            "method": "Coordinator Conjugate Gradient",
            "device": "mps",
            "grid": "100x100",
            "time_sec": coordinator_time,
            "iterations": coordinator_iterations,
            "laplace_residual": coordinator_residual,
            "peak_process_memory_mib": (
                f"{coordinator_memory_mib:.2f}"
                if coordinator_memory_mib is not None
                else "n/a"
            ),
        },
    ]

    csv_path = write_csv(rows)
    markdown_path = write_markdown_report(
        direct_output=direct_output,
        coordinator_output=coordinator_output,
        rows=rows,
        max_difference=max_difference,
        rms_difference=rms_difference,
    )

    print("\n=== Benchmark summary ===")
    for row in rows:
        print(
            f"{row['method']}: "
            f"time={row['time_sec']} sec, "
            f"iterations={row['iterations']}, "
            f"residual={row['laplace_residual']}, "
            f"peak RAM={row['peak_process_memory_mib']} MiB"
        )

    print(f"Max |coordinator - direct|: {max_difference}")
    print(f"RMS |coordinator - direct|: {rms_difference}")
    print(f"CSV saved: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report saved: {markdown_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()