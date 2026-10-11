# Test entry points

| File/directory | Purpose | How it runs |
|---|---|---|
| `runtests.jl` | Current problem/FastNCG correctness checks and ANCG core tests | `julia --project=. test/runtests.jl` |
| `ancg_reference.jl` | ANCG CG/outer-flow branch tests | Included by runtests; can also run separately |
| `ancg_adapter.jl` | ANCG interface and oracle accounting | `julia --project=. test/ancg_adapter.jl` |
| `test.jl` | Deterministic numerical regression against the current baseline | `julia --project=. test/test.jl`; not included by runtests |
| `ancg_cross_language/` | Julia/MATLAB ANCG validation | See its README |
| `arncg_cross_language/` | Julia/MATLAB ARNCG validation and ARCqK checks | See its README |

## Current regression baseline, 2026-10-10

The baseline in `test.jl` was refreshed for the current implementation using
Julia 1.12.6 and OpenBLAS ILP64. A separate run of `julia --project=.
test/test.jl` returned exit code **0**, with **PASS for all four problems**.
See the [repository README](../README.md#tests-and-regression) for the result table;
the script stores the full-precision reference values.

The script sets BLAS to one thread and restores the prior setting in `finally`.
It retains exact iteration/oracle counts and the existing value tolerances
(`rtol=5e-5`, `atol=1e-14`). Additional checks require first-order convergence,
final gradient norm at most `1e-7`, finite final iterates and loss/gradient
histories, and nonincreasing loss. Seven targeted checks of the pass/fail
predicate also passed, including rejection of a failure status, an excessive
gradient norm, nonfinite data, increasing loss, and an incorrect oracle count.
These targeted checks were run separately and are not part of the test entry points.

The solver algorithm, problem definitions, solver settings, and comparison
tolerances were not changed. This script deliberately omits `objdiff_fn` and
therefore exercises a different interface configuration from the benchmark.
Exact trajectories can still vary across Julia/BLAS versions and hardware.
The complementary correctness suite `runtests.jl` passed 291 assertions before
this test-only baseline refresh.

## Before the baseline refresh

The initial run against the older references failed all four comparisons even
though all four final gradient norms were at most `1e-7`. The old/current counts
were:

| Problem | Iter old / current | HVP old / current | Obj old / current | Grad old / current |
|---|---:|---:|---:|---:|
| Rosenbrock | 2517 / 2517 | 15148 / 15149 | 7566 / 7567 | 7566 / 7567 |
| LogSumExp | 54 / 55 | 419 / 425 | 206 / 196 | 206 / 196 |
| Polytope | 51 / 47 | 98 / 79 | 112 / 89 | 112 / 89 |
| TRegression | 193 / 173 | 1230 / 1201 | 708 / 646 | 708 / 646 |

The old TRegression loss was `1390.4074119109016`; the current baseline is
`1416.0209426997492`. First-order convergence does not establish the same local
solution or improved objective quality. The refreshed references record current
behavior for future regression detection; they do not establish numerical
equivalence with the old implementation.
