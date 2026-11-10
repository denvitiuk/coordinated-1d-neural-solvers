# Laplace on rectangles: coordinating two families of 1D systems

Top boundary: `sin` = sin(πx/Lx) (one harmonic), `const` = 1 (all harmonics). float64, CPU, rtol = 1e-10, max_iter = 3000.
Solves = batched tridiagonal solves (one pass over all rows or all columns). Error = max |U − U*| vs the exact solution of the same 5-point scheme.

| Top | Grid (x·y) | Domain | Method | Iterations | Solves | Time, s | Error | Converged |
|---|---|---|---|---|---|---|---|---|
| sin | 100x100 | 1x1 | SD | 34 | 138 | 0.146 | 1.4e-10 | yes |
| sin | 100x100 | 1x1 | CGLS | 9 | 38 | 0.037 | 1.9e-12 | yes |
| sin | 100x100 | 1x1 | CR | 8 | 18 | 0.018 | 9.3e-13 | yes |
| sin | 100x100 | 1x1 | GMRES | 8 | 18 | 0.020 | 9.3e-13 | yes |
| sin | 121x61 | 2x1 | SD | 15 | 62 | 0.056 | 3.5e-11 | yes |
| sin | 121x61 | 2x1 | CGLS | 7 | 30 | 0.027 | 1.1e-12 | yes |
| sin | 121x61 | 2x1 | CR | 6 | 14 | 0.013 | 4.1e-12 | yes |
| sin | 121x61 | 2x1 | GMRES | 6 | 14 | 0.013 | 4.1e-12 | yes |
| sin | 101x41 | 2x1 | SD | 15 | 62 | 0.043 | 2.9e-11 | yes |
| sin | 101x41 | 2x1 | CGLS | 7 | 30 | 0.021 | 1.2e-12 | yes |
| sin | 101x41 | 2x1 | CR | 6 | 14 | 0.010 | 3.4e-12 | yes |
| sin | 101x41 | 2x1 | GMRES | 6 | 14 | 0.010 | 3.4e-12 | yes |
| sin | 41x121 | 1x3 | SD | 301 | 1206 | 0.952 | 1.9e-09 | yes |
| sin | 41x121 | 1x3 | CGLS | 22 | 90 | 0.070 | 1.1e-10 | yes |
| sin | 41x121 | 1x3 | CR | 13 | 28 | 0.022 | 6.1e-11 | yes |
| sin | 41x121 | 1x3 | GMRES | 12 | 26 | 0.022 | 6.9e-11 | yes |
| const | 100x100 | 1x1 | SD | 3000 | 12002 | 12.070 | 1.4e-01 | no |
| const | 100x100 | 1x1 | CGLS | 3000 | 12002 | 11.929 | 7.1e-07 | no |
| const | 100x100 | 1x1 | CR | 291 | 584 | 0.579 | 6.3e-11 | yes |
| const | 100x100 | 1x1 | GMRES | 294 | 594 | 0.798 | 8.9e-11 | yes |
| const | 121x61 | 2x1 | SD | 3000 | 12002 | 11.054 | 1.4e-01 | no |
| const | 121x61 | 2x1 | CGLS | 3000 | 12002 | 10.640 | 3.2e-06 | no |
| const | 121x61 | 2x1 | CR | 315 | 632 | 0.563 | 5.6e-11 | yes |
| const | 121x61 | 2x1 | GMRES | 320 | 648 | 0.794 | 7.4e-11 | yes |
| const | 101x41 | 2x1 | SD | 3000 | 12002 | 8.216 | 1.2e-01 | no |
| const | 101x41 | 2x1 | CGLS | 3000 | 12002 | 8.142 | 1.6e-08 | no |
| const | 101x41 | 2x1 | CR | 239 | 480 | 0.330 | 5.8e-11 | yes |
| const | 101x41 | 2x1 | GMRES | 236 | 478 | 0.475 | 9.2e-11 | yes |
| const | 41x121 | 1x3 | SD | 3000 | 12002 | 9.483 | 1.3e-01 | no |
| const | 41x121 | 1x3 | CGLS | 3000 | 12002 | 9.358 | 4.6e-07 | no |
| const | 41x121 | 1x3 | CR | 283 | 568 | 0.448 | 7.3e-11 | yes |
| const | 41x121 | 1x3 | GMRES | 282 | 570 | 0.629 | 6.5e-11 | yes |
