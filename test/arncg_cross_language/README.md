# ARNCG MATLAB / Julia checks

Run from the repository root:

```sh
julia --project=. test/arncg_cross_language/check.jl /tmp/srncg-arncg-check
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('test/arncg_cross_language'); run_reference('/tmp/srncg-arncg-check')"
python3 test/arncg_cross_language/compare.py /tmp/srncg-arncg-check
```

The Julia check includes a convergence/HVP-count smoke test of ARCqK with the
paper configuration. The MATLAB comparison applies only to ARNCG and runs the
unmodified pinned reference with the same inputs, 1e-7 tolerance, and one BLAS
thread. HVPs are counted by a callback wrapper. Iterations count MATLAB records.

Observed with MATLAB R2026a and Julia 1.12.6:

| Fixture | Iterations (both) | HVP (both) | Maximum scaled loss/gradient history difference |
|---|---:|---:|---:|
| Initially stationary | 0 | 0 | 0 |
| Positive-definite quadratic | 5 | 20 | 1.148e-16 |
| Rosenbrock, 2 variables | 24 | 98 | 5.940e-13 |
| Quartic with negative curvature | 7 | 28 | 0 |

Objective/gradient/HVP counts, alpha, M, CG iterations, and direction types match
in all four cases. Final x, loss and gradient meet the comparison tolerance.
Scaled error is `abs(a-b)/(1+max(abs(a),abs(b)))`.
The Julia checks pass 16 assertions; they explicitly require a negative-curvature
step on the quartic fixture. Cross-language roundoff can still change decisions
on other problems. These fixtures do not exercise every exceptional branch.

An additional smoke run through `run_method` on the benchmark's 400-variable
Polytope problem reached the shared 1e-7 tolerance:

| Method | Iterations | HVP | Loss | Gradient norm |
|---|---:|---:|---:|---:|
| ARCqK, paper configuration | 19 | 543 | 2.1524470917695342e-10 | 8.069036155481114e-8 |
| ARNCG | 19 | 1795 | 5.654744465960822e-10 | 6.686712094041499e-8 |

This Polytope smoke run was Julia-only; it is not a MATLAB parity result or a
timing benchmark. ARNCG's change from the previous reported 20 iterations to 19
corrects the final convergence-check overcount, not the numerical algorithm.
