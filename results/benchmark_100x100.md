# Benchmark: equation of Laplace on a 100 × 100 grid

## Methods

1. **Direct CPU method**: Jacobi iteration with the 5-point finite-difference stencil.
2. **Coordinator method**: 98 horizontal and 98 vertical tridiagonal systems, solved in batches; coordination uses conjugate gradients.

## Results

| Metric | Direct CPU Jacobi | Coordinator MPS |
|---|---:|---:|
| Time, sec | 0.3620 | 0.9571 |
| Iterations | 14654 | 57 |
| Laplace residual | 9.99997916e-08 | 7.15255737e-07 |
| Peak process memory, MiB | 32.41 | 260.73 |

## Agreement of final solutions

- Max |coordinator − direct|: `1.98519224e-04`
- RMS |coordinator − direct|: `9.85735428e-05`

## Notes

- Peak process memory is measured by macOS `/usr/bin/time -l` and reflects process RAM. It is not a direct measurement of separate NVIDIA VRAM.
- For this 100 × 100 task, MPS overhead may dominate runtime. The coordinator method's parallel structure is the batched solution of 98 horizontal and 98 vertical independent tridiagonal systems at each conjugate-gradient iteration.

## Raw output: direct CPU solver

```text
=== Direct 2D Laplace solver: CPU ===
Grid: 100 x 100
Iterations: 14654
Final error: 9.99997916e-08
Time: 0.3620 sec
Min value: 0.000000
Max value: 0.999874
Saved: direct_laplace_solution.npy
```

## Raw output: coordinator solver

```text
Iteration 020 | CG residual norm: 4.59206649e-06
Iteration 040 | CG residual norm: 2.91919059e-06
Iteration 057 | CG residual norm: 6.82926327e-07

=== Coordinator Laplace solver: Conjugate Gradient ===
Device: mps
Grid: 100 x 100
CG iterations: 57
CG residual norm: 6.82926327e-07
Max |u - v|: 6.25848770e-07
RMS |u - v|: 1.10067255e-07
Laplace residual: 7.15255737e-07
Time: 0.9571 sec
Saved: coordinator_laplace_cg_solution.npy

=== Comparison with direct CPU solution ===
Max |coordinator - direct|: 1.98519224e-04
RMS |coordinator - direct|: 9.85735428e-05
```
