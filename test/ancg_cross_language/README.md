# MATLAB / Julia ANCG cross-language check

This harness executes the pinned, unmodified MATLAB files and the Julia reference
with the same input arrays and starting points. All mathematical options are
specified; both runs use one BLAS thread and explicit analytic Hessians. It
checks the reference core independently of the benchmark's HVP-to-Hessian adapter.

Run from the repository root (substitute your MATLAB executable):

```sh
julia --project=. test/ancg_cross_language/prepare.jl /tmp/srncg-ancg-cross-language
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('test/ancg_cross_language'); run_reference('/tmp/srncg-ancg-cross-language')"
python3 test/ancg_cross_language/compare.py /tmp/srncg-ancg-cross-language test/ancg_cross_language/RESULTS.md
```

`prepare.jl` writes shared A/b/x0 files and Julia outputs; `run_reference.m`
reads them and writes MATLAB outputs. `compare.py` reports every history/counter
difference rather than requiring or claiming bitwise equality. It asserts exact
branch/counter agreement and tight numerical agreement on analytic fixtures,
and checks convergence of both implementations on the four nonlinear problems.
It also checks the original CappedCG initial negative-curvature fall-through
and reproduces the source's uninitialized warm-start variable in MATLAB.

Cases: scalar quadratic, SOL/NC line-search failures, Rosenbrock (100 variables),
Polytope (400 variables, 500 rows, power 4), Student-t (100 variables, 200 rows,
nu=0.1), and LogSumExp (200 variables, 1000 rows, rho=0.05).

The fixture callbacks deliberately compute explicit Hessians directly. Thus
these traces may differ from the benchmark adapter's reconstructed-Hessian runs
through floating-point rounding, even within Julia. The input data is shared
through 17-digit decimal files to avoid differences between Julia and MATLAB
random-number generators.

See [RESULTS.md](RESULTS.md) for the observed R2026a comparison. Matching formulas
and control flow does not guarantee identical floating-point trajectories,
especially for CappedCG stopping tests and objective comparisons near convergence.

See [NUMERICAL_NOTES.md](NUMERICAL_NOTES.md) for same-input replay experiments
on the first differing CappedCG calls in Student-t and LogSumExp.
