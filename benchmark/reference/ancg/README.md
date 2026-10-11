# Pinned ANCG reference

Upstream: https://github.com/Zeng-Ziyang/ANCG-Holder-Hessian

Commit: `45532f314fbf2f0c064f676c64bb9e53db0c6ee6`. The MATLAB files in this directory are unmodified downloads
from this commit; they are the reference for `../../../src/ANCG.jl`.

SHA-256:

- `ANCG.m`: `6f5b115d6f8b25d47d33ea050c150aa0e512aaf9625fa71fd7a500456606905e`
- `capCG.m`: `96ed05b3f5ab2ba871937aabc4c59571df8846c287e3edd71cddc358cc9359e5`

## What is preserved

The Julia reference takes objective, gradient, and explicit-Hessian callbacks,
as the authors' MATLAB code does. It preserves default mathematical parameters,
CG evaluation order and HVP accounting, all three curvature-bound updates,
residual-growth decay and historical-direction search, inclusive backtracking,
gamma caps/restarts, acceptance tests, stop reasons, and all output histories.
No extra CG iteration limit, small-gradient step, denominator floor of 1e-15,
stable objective-difference callback, or exception recovery is added.
`floatmin(Float64)` is the translation of MATLAB `realmin`.

Known source behavior is deliberately retained:

- The initial negative-curvature assignment in `capCG.m` does not return;
  execution continues into the CG loop.
- The optional warm-start branch does not initialize `j`; it fails in the
  source and raises an uninitialized-variable error in the Julia translation.
  The default `warmstartCG=false` is usable.
- If the historical-direction search fails to set all return values, the
  reference does not invent a fallback; it can raise an uninitialized-output
  error, as can the source.
- There is no CG iteration cap. MATLAB's automatic history-matrix growth is
  represented by explicitly adding columns when necessary.

Language/interface adaptations are explicit:

- Julia keyword defaults supply all options. The MATLAB `get`/`iff` helper
  eagerly reads missing fields, so running the original requires a fully
  populated options struct. The numerical options themselves are unchanged.
- MATLAB's scalar-zero initial CG vector is represented as a zero vector.
- `eps_tol` names MATLAB's `eps` option. Process CPU time uses the platform C
  `clock()` units; the benchmark independently measures wall time.
- Cross-language BLAS/rounding differences can still change boundary decisions.
  MATLAB R2026a comparisons are recorded in
  [the cross-language results](../../../test/ancg_cross_language/RESULTS.md).
  Not every problem has identical iteration or HVP counts.

## Benchmark adapter and cost accounting

`../../../src/ANCG.jl` retains the public `ancg_optimize` entry point. Its
`max_iter`, `tol`, `gamma_0`, and `max_ls` keywords map to the reference options
`maxit`, `eps_tol`, `gamma0`, and `maxls`. Benchmark printing is disabled; the
reference function itself defaults to verbose output as the MATLAB source does.
The common benchmark tolerance and iteration limit override the source defaults.

Existing problem factories supply HVPs, not full Hessians. The adapter therefore
constructs each Hessian using one HVP per column, **inside the measured solve**.
It does not symmetrize or otherwise alter the assembled matrix. An analytic
Hessian callback can instead be supplied with `hess_fn`.

- `result.details.hvp_count`: authors' CG matrix-vector product count.
- `result.details.Hevals`: authors' full-Hessian evaluation count.
- `result.details.hessian_hvp`: HVP calls used to construct those Hessians.
- `result.hvp`: sum of CG products and Hessian-construction HVPs.
- `result.details.fx_hist`, `gn_hist`, `gamma_hist`, `alpha_hist`, `dtype_hist`,
  and `stop`: authors' histories and termination reason.

Thus the benchmark includes the explicit-Hessian adapter's cost; its timings
must not be presented as the authors' native MATLAB timings or as a matrix-free
ANCG implementation. The NLPModels HVP counter measures reconstruction calls;
CG uses the assembled matrix directly and is counted separately.

## Validation

Run `julia --project=. test/ancg_reference.jl` for analytic and branch tests,
and `julia --project=. test/ancg_adapter.jl` for adapter/counter checks.
The subsequent MATLAB R2026a comparison uses shared input arrays and explicit
analytic Hessians. Analytic fixtures agree exactly; Rosenbrock and Polytope
match all recorded counts and step choices. Student-t and LogSumExp have some
trajectory/counter differences, despite both reaching the same gradient tolerance.
See the [reproducible cross-language harness](../../../test/ancg_cross_language/README.md)
and its full results. No original MATLAB arithmetic or Julia solver logic was
changed to force matching trajectories.
