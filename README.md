# FastNCG.jl

The local package/module is now `FastNCG` and the solver type is
`FastNCGSolver`. Restart Julia after migrating, activate this project, then use:

```julia
using FastNCG
solver = FastNCGSolver()
```

Julia cannot define a type named `FastNCG` inside the module `FastNCG`, so the
qualified constructor is `FastNCG.FastNCGSolver()`. The package UUID is unchanged.
The existing GitHub repository URL below is retained; this local rename does not
rename or publish that remote repository. Until the changes are published, use
the local checkout with `julia --project=.` or `Pkg.develop(path="/path/to/checkout")`.

**FastNCG.jl** is a Julia implementation of **FastNCG**, a matrix-free second-order optimization method based on regularized Newton steps and adaptive Krylov subspace approximations.

The method is designed for large-scale smooth optimization problems where explicitly forming or factorizing the Hessian is expensive or impractical. Instead of constructing the full Hessian matrix, FastNCG accesses second-order information through **Hessian-vector products (HVPs)** and constructs a low-dimensional Lanczos/Krylov approximation of the regularized Newton system.

The package is built on top of [`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and can be used either with an `NLPModels`-compatible optimization model or with the lightweight `ClosureNLPModel` interface provided in this repository.

---

## Features

FastNCG.jl provides:

- matrix-free second-order optimization;
- Hessian-vector-product-based computation;
- adaptive Lanczos/Krylov subspace construction;
- regularized Newton steps;
- reduced-dimensional subproblem solves;
- adaptive regularization;
- compatibility with `NLPModels.jl`;
- a lightweight closure-based interface for custom objectives;
- detailed solver statistics and optimization histories;
- deterministic regression tests;
- benchmark scripts for comparison with several first- and second-order optimization methods.

The core algorithm does **not** require explicit construction or factorization of the Hessian.

---

## Requirements

The current implementation is developed with Julia 1.12.

The main dependencies are:

- `NLPModels.jl`
- `LinearAlgebra`

The repository also includes dependencies used by the regression tests and benchmark scripts:

- `AdaptiveRegularization.jl`
- `BenchmarkTools.jl`
- `TimerOutputs.jl`
- `Random`
- `Printf`

All dependencies are specified in `Project.toml`.

---

## Installation

### Install directly from GitHub

Once the repository is public, the package can be installed with

```julia
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SRNCG.git")
```

Then load the package with

```julia
using FastNCG
```

If the repository name or GitHub username is different, replace the URL accordingly.

### Development installation

If you want to modify the source code, clone the repository and activate it locally:

```bash
git clone https://github.com/Xunjian-Li/SRNCG.git FastNCG
cd FastNCG
```

Start Julia with the project environment:

```bash
julia --project=.
```

Then instantiate the dependencies:

```julia
using Pkg
Pkg.instantiate()
```

and load FastNCG:

```julia
using FastNCG
```

Alternatively, from another Julia environment you can use

```julia
using Pkg
Pkg.develop(path="/path/to/FastNCG")
```

---

# Quick Start

The basic workflow is

```julia
using FastNCG

solver = FastNCGSolver()

x = optimize!(
    solver,
    nlp;
    initial_theta=x0,
    tol=1e-7,
    max_iter=10_000,
)
```

where:

- `nlp` is an `NLPModels.jl`-compatible model;
- `x0` is the initial point;
- `tol` is the first-order stopping tolerance;
- `max_iter` is the maximum number of outer iterations.

The returned value `x` is the final iterate.

---

# Defining a Problem

FastNCG requires three basic operations:

1. objective evaluation
   \[
   f(x),
   \]

2. gradient evaluation
   \[
   \nabla f(x),
   \]

3. Hessian-vector product
   \[
   \nabla^2 f(x)v.
   \]

The full Hessian matrix does not need to be formed.

FastNCG.jl provides `ClosureNLPModel` to conveniently wrap these three operations into an `NLPModels` model.

---

## Example: Quadratic Optimization

Consider

\[
f(x)=\frac12 x^\top A x-b^\top x,
\]

where \(A\) is symmetric positive definite.

Its gradient and Hessian-vector product are

\[
\nabla f(x)=Ax-b,
\]

and

\[
\nabla^2 f(x)v=Av.
\]

A complete FastNCG example is:

```julia
using FastNCG
using LinearAlgebra
using Random

Random.seed!(1234)

n = 100

A = randn(n, n)
A = A' * A + I

b = randn(n)
x0 = zeros(n)

function obj_fn(x)
    return 0.5 * dot(x, A * x) - dot(b, x)
end

function grad_fn!(g, x)
    mul!(g, A, x)
    g .-= b
    return g
end

function hprod_fn!(Hv, x, v)
    mul!(Hv, A, v)
    return Hv
end

nlp = ClosureNLPModel(
    copy(x0),
    obj_fn,
    grad_fn!,
    hprod_fn!;
    name="Quadratic",
)

solver = FastNCGSolver(max_outer=1000)

x = optimize!(
    solver,
    nlp;
    initial_theta=copy(x0),
    max_iter=1000,
    tol=1e-7,
)

println("Final objective      = ", solver.loss_history[end])
println("Final gradient norm  = ", solver.grad_norm_history[end])
println("HVP evaluations      = ", solver.hvp_count)
println("Total time           = ", solver.total_time)
```

For this example, the exact solution is

```julia
x_exact = A \ b
```

so the numerical solution can be checked using

```julia
norm(x - x_exact) / norm(x_exact)
```

---

# `ClosureNLPModel`

The constructor is

```julia
ClosureNLPModel(
    x0,
    obj_fn,
    grad_fn,
    hprod_fn;
    name="Closure",
)
```

The callbacks must have the following forms.

### Objective

```julia
obj_fn(x)
```

returns the scalar objective value.

### Gradient

```julia
grad_fn(g, x)
```

writes the gradient into `g`.

For example,

```julia
function grad_fn!(g, x)
    # compute gradient
    return g
end
```

### Hessian-vector product

```julia
hprod_fn(Hv, x, v)
```

writes

\[
Hv=\nabla^2f(x)v
\]

into `Hv`.

For example,

```julia
function hprod_fn!(Hv, x, v)
    # compute Hessian-vector product
    return Hv
end
```

For large-scale applications, `hprod_fn` should preferably compute the HVP directly without explicitly constructing the Hessian.

---

# Solver Interface

Create a solver with

```julia
solver = FastNCGSolver()
```

The current constructor is

```julia
FastNCGSolver(
    sigma_0_ref = 1.0,
    sigma_min   = 1e-8,
    rho         = 2.0,
    beta        = 0.25,
    gamma       = 1e-3,
    tau         = 0.1,
    max_outer   = 500,
    max_krylov  = 0,
    max_newton  = 10,
    max_regularization_retries = 50,
    tol_newton  = 1e-3,
    verbose     = false,
)
```

The optimization routine is called as

```julia
x = optimize!(
    solver,
    nlp;
    initial_theta=x0,
    max_iter=1000,
    tol=1e-7,
)
```

`max_regularization_retries` limits increases of `sigma` per outer iteration,
independently of Krylov dimension. A value of zero permits the initial `sigma`
only. Krylov extensions do not consume this retry budget.

After a solve, `solver.status` is `:first_order` when the gradient tolerance is
met, `:max_iter` when the outer iteration limit is reached, or `:inner_failure`
when no acceptable step is found within the regularization retry budget.
`:numerical_stagnation` indicates that adding the computed step leaves every
coordinate unchanged in floating-point arithmetic; no trial callbacks are
evaluated in this case. It does not imply convergence.
`optimize!` continues to return the solution vector. If `initial_theta` is
omitted, it starts from a copy of `nlp.meta.x0` without modifying the model's
saved initial point.

The objective-difference callback is evaluated only after the projection and
gradient conditions pass. When a finite `objdiff_fn` result is available, it supplies the stable objective
difference for the sufficient-decrease test. Otherwise, a trial must show a
strictly negative computed objective difference and satisfy sufficient decrease.
A quadratic model prediction cannot substitute for observed decrease. For
objectives with a large constant offset or very small decreases near convergence,
provide `objdiff_fn`; rounded-to-zero differences are rejected.

Krylov basis storage starts with at most 17 columns and grows as needed, up to
`max_krylov + 1`. The maximum allowed Krylov dimension is unchanged.

For example, verbose output can be enabled with

```julia
solver = FastNCGSolver(verbose=true)
```

---

# Solver Statistics

After calling `optimize!`, computational statistics are stored in the solver object.

Important fields include

```julia
solver.hvp_count
solver.grad_count
solver.func_count

solver.krylov_extensions
solver.ratio_increases
solver.newton_iters_total

solver.accepted_kappa
solver.attained_dims

solver.loss_history
solver.grad_norm_history
solver.time_history
solver.total_time
```

For example:

```julia
println("Iterations          = ", length(solver.loss_history) - 1)
println("HVP evaluations     = ", solver.hvp_count)
println("Objective evals     = ", solver.func_count)
println("Gradient evals      = ", solver.grad_count)
println("Krylov extensions   = ", solver.krylov_extensions)
println("Total time          = ", solver.total_time)

println("Final objective     = ", solver.loss_history[end])
println("Final gradient norm = ", solver.grad_norm_history[end])
```

The histories can also be used to inspect convergence:

```julia
f_history = solver.loss_history
g_history = solver.grad_norm_history
t_history = solver.time_history
```

---

# Included Test Problems

The repository contains several smooth optimization problems used for testing and benchmarking:

- **Rosenbrock**
- **Log-Sum-Exp**
- **Polytope**
- **T-Regression**

They are available through

```julia
make_rosenbrock
make_logsumexp
make_polytope
make_tregression
```

For example:

```julia
using FastNCG

x0, obj_fn, grad_fn!, hprod_fn! =
    make_rosenbrock(1000; x0=zeros(1000))

nlp = ClosureNLPModel(
    copy(x0),
    obj_fn,
    grad_fn!,
    hprod_fn!;
    name="Rosenbrock",
)

solver = FastNCGSolver(max_outer=10_000)

x = optimize!(
    solver,
    nlp;
    initial_theta=copy(x0),
    max_iter=10_000,
    tol=1e-7,
)
```

---

# Tests and Regression

Run the tests from the repository root:

```bash
julia --project=. test/runtests.jl
julia --project=. test/ancg_adapter.jl
julia --project=. test/test.jl
```

`runtests.jl` covers problem derivatives/caches, FastNCG acceptance and stopping
behavior, workspace operations, and the ANCG reference branches. The separate
adapter test checks ANCG model integration and evaluation counters.

`test/test.jl` compares four deterministic runs against the current numerical
baseline, refreshed on 2026-10-10 using Julia 1.12.6 and OpenBLAS ILP64. It fixes
BLAS to one thread during the test and restores the previous setting afterward.
The refreshed baseline was verified in a separate run, which exited successfully:

| Problem | Parameters | Iter | HVP | Obj | Grad | Final loss | Gradient norm | Regression |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Rosenbrock | d=1000 | 2517 | 15149 | 7567 | 7567 | 1.4715e-22 | 4.1313e-10 | PASS |
| LogSumExp | d=200, n=1000, rho=0.05 | 55 | 425 | 196 | 196 | 1.4623e+00 | 1.5021e-08 | PASS |
| Polytope | d=400, m=500, p=4 | 47 | 79 | 89 | 89 | 1.4004e-09 | 9.8264e-08 | PASS |
| TRegression | d=200, m=500, nu=0.001 | 173 | 1201 | 646 | 646 | 1.4160e+03 | 7.4288e-08 | PASS |

Iteration and oracle counts must still match exactly; numerical comparison
tolerances remain `rtol=5e-5`, `atol=1e-14`. The test additionally requires
`:first_order` status, a final gradient norm at most `1e-7`, finite final iterates
and objective/gradient histories, and nonincreasing loss.

The solver algorithm and problem/solver configurations were not changed by this
baseline refresh. Passing verifies agreement with the current baseline, not the
older implementation. In particular, TRegression's old loss was approximately
1390.4, versus 1416.0 now; this refresh does not establish improved objective
quality or agreement with the old solution.

This script deliberately tests only the objective, gradient and HVP callbacks;
it omits `objdiff_fn`, unlike the benchmark. Its trajectory can also depend on
Julia/BLAS versions and hardware despite the fixed thread count.
See [test/README.md](test/README.md) for test roles and baseline history.

---

# Benchmark

A benchmark script is included in

```text
benchmark/benchmark.jl
```

The current benchmark compares

- **ARCqK**
- **FastNCG**
- **Gradient Descent (GD)**
- **Nesterov Accelerated Gradient (NAG)**
- **ANCG**
- **ARNCG**

on the same four test problems. FastNCG is the benchmark name for the solver
constructed with `FastNCG.FastNCGSolver`. The package/module is `FastNCG`;
the solver type is `FastNCGSolver` because Julia modules and types cannot share
a binding in the same namespace.

The benchmark uses a common first-order tolerance

\[
\|\nabla f(x)\|\le 10^{-7},
\]

with a maximum of 10,000 outer iterations.

Wall-clock timing is performed using `BenchmarkTools.jl`.

Each benchmark evaluation corresponds to one complete optimization run:

```text
evals = 1
```

and the default benchmark duration is

```text
5 seconds per method/problem
```

Problem construction and mutable cache initialization are performed outside the timed expression.

Algorithmic counters and final objective/gradient values are obtained from a separate deterministic run.

This separates:

1. **wall-clock performance**, measured using `BenchmarkTools`, and
2. **algorithmic work**, measured using iterations and oracle evaluations.

---

## Running the Benchmark

From Julia:

```julia
using Pkg
Pkg.activate(".")

using FastNCG

include("benchmark/benchmark.jl")
```

or directly from the command line:

```bash
julia --project=. benchmark/benchmark.jl
```

The output contains columns of the form

```text
Problem
Method
Median(s)
Q25(s)
Q75(s)
Samples
Iter
HVP
Obj
Grad
f(x*)
|g(x*)|
Status
```

For example, timing is summarized using the median and interquartile range rather than a single wall-clock measurement.

---

# Benchmark Methodology

When comparing second-order methods, wall-clock time alone can be implementation-dependent. The benchmark therefore reports several measures of computational work.

### Iter

Number of outer iterations.

### HVP

Number of Hessian-vector products

\[
v\mapsto\nabla^2f(x)v.
\]

This is particularly important for matrix-free second-order algorithms.

### Obj

Number of objective evaluations.

### Grad

Number of gradient evaluations.

### Final objective

```text
f(x*)
```

### First-order stationarity

```text
|g(x*)|
```

denotes

\[
\|\nabla f(x^\ast)\|.
\]

A run is reported as converged when the final gradient norm satisfies the benchmark tolerance.

---

# Notes on Benchmark Baselines

The benchmark contains implementations of several comparison methods.

The ANCG implementation is a source-faithful Julia translation of the authors'
`ANCG.m` and `capCG.m`, pinned to commit
`45532f314fbf2f0c064f676c64bb9e53db0c6ee6`. It uses explicit Hessians, including
in the benchmark adapter. Source provenance, preserved upstream quirks, parameter
mapping, and validation limits are documented in
[the ANCG reference notes](benchmark/reference/ancg/README.md).

For ANCG, the summary HVP count includes both CG matrix-vector products and HVPs
used to reconstruct Hessians from the problem interface. Their separate counts,
all loss/gradient/parameter histories, and the original stop reason are available
in `rows[i].details`. The reported runtime includes reconstruction. This is not
a matrix-free implementation, nor a measurement of native MATLAB performance.

Run `julia --project=. benchmark/benchmark.jl` to execute the full benchmark.
When loading it with `include`, call `rows = main()` explicitly.

The ARNCG implementation follows the authors' MATLAB `recompute` implementation,
pinned to commit `8605f1e13386f3d0908a3d4a6dce9ce555d890bf`. The default is their
experimental ARNCG_g configuration with theta=1 and lambda=0 (fallback disabled).
Iterations count executed outer steps, including rejected steps, and HVP counts
include every actual product. See the [reference notes](benchmark/reference/arncg/README.md)
and [MATLAB comparison](test/arncg_cross_language/README.md) for preserved source
behavior and validation limits.

ARCqK uses `AdaptiveRegularization.jl` 0.2.2 with the settings from its
`paper/benchmark_arctr.jl`: shifts `10 .^ (-10:0.5:20)`, zeta=0.5, initial
alpha=10, acceptance thresholds 0.1/0.75, and decrease/increase factors 0.1/5.
The old `TrustRegion` constructor in that reproduction script maps to the
current `ARTrustRegion` constructor. The benchmark retains the shared absolute
gradient tolerance 1e-7 (relative tolerance zero), and its ARCqK time budget of
120 seconds. Thus this compares the paper's method/configuration on our problems;
it does not reproduce the paper's stopping tolerances, time budget, or tables.

The comparison algorithms live in `src` and are exported by FastNCG.
The benchmark script handles problem construction, scheduling, timing and reporting.

Users interested only in FastNCG do not need to use these benchmark routines.

---

# Repository Structure

The current repository is organized as

```text
FastNCG/
├── Project.toml
├── Manifest.toml
├── test/
│   ├── runtests.jl
│   └── test.jl
├── benchmark/
│   └── benchmark.jl
├── README.md
└── src/
    ├── FastNCG.jl
    ├── Fast_NCG.jl
    └── problems.jl
```

The files have the following roles.

### `src/FastNCG.jl`

Main package module.

It loads the FastNCG implementation and the included test-problem constructors.

### `src/Fast_NCG.jl`

Core FastNCG implementation.

This file contains:

- `FastNCGSolver`
- `ClosureNLPModel`
- `optimize!`
- Lanczos/Krylov operations
- reduced regularized subproblem calculations
- solver statistics and histories

### `src/problems.jl`

Test-problem constructors:

```julia
make_rosenbrock
make_logsumexp
make_polytope
make_tregression
```

### `test/test.jl`

Deterministic numerical regression against the baseline refreshed on 2026-10-10.
All four problems pass; use `test/runtests.jl` for the complementary correctness suite.

### `benchmark/benchmark.jl`

Benchmark suite comparing FastNCG with other optimization methods.

---

# Using FastNCG in Your Own Code

For most users, only three objects are needed:

```julia
using FastNCG

ClosureNLPModel
FastNCGSolver
optimize!
```

A typical application therefore has the following structure:

```julia
using FastNCG

# 1. Define objective
function f(x)
    # ...
end

# 2. Define gradient
function grad_f!(g, x)
    # ...
    return g
end

# 3. Define Hessian-vector product
function Hv_f!(Hv, x, v)
    # ...
    return Hv
end

# 4. Initial point
x0 = ...

# 5. Build NLPModels-compatible model
nlp = ClosureNLPModel(
    copy(x0),
    f,
    grad_f!,
    Hv_f!;
    name="MyProblem",
)

# 6. Construct FastNCG solver
solver = FastNCGSolver()

# 7. Optimize
x = optimize!(
    solver,
    nlp;
    initial_theta=copy(x0),
    tol=1e-7,
    max_iter=10_000,
)

# 8. Inspect the result
println("f(x)   = ", solver.loss_history[end])
println("|g(x)| = ", solver.grad_norm_history[end])
println("HVPs   = ", solver.hvp_count)
```

The main computational requirement is therefore an efficient implementation of

```julia
Hv_f!(Hv, x, v)
```

rather than an explicit Hessian matrix.

This is the intended use case of FastNCG for large-scale problems.

---

# Reproducibility

The supplied stochastic test problems use fixed random seeds where applicable.

The `test/test.jl` regression checks quantities such as:

- iteration counts;
- Hessian-vector-product counts;
- objective evaluation counts;
- gradient evaluation counts;
- final objective values;
- final gradient norms.

Wall-clock timings are not used as regression criteria because they depend on hardware, Julia version, BLAS configuration, and system load.

For reliable benchmark timing, it is recommended to run the benchmark in a fresh Julia session.

---

# Citation

If you use FastNCG.jl in academic work, please cite the accompanying FastNCG paper.

The full citation will be added when the manuscript is publicly available.

A BibTeX entry will be provided here:

```bibtex
@article{FastNCG,
  title   = {FastNCG},
  author  = {Xun-Jian Li and coauthors},
  journal = {...},
  year    = {...}
}
```

---

# Author

**Xun-Jian Li**

Department of Biostatistics  
University of California, Los Angeles

---

# License

MIT License

Copyright (c) 2026 Xun-Jian Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Algorithm source layout

- `src/Fast_NCG.jl`: FastNCGSolver and the closure-model interface.
- `src/GD.jl`, `src/NAG.jl`, `src/CR.jl`: respective comparison solvers.
- `src/ANCG.jl`: ANCG reference core and model adapter in one file.
- `src/ARNCG.jl`: authors' ARNCG translation.
- `src/ARCqK.jl`: ARCqK package adapter and paper configuration.
- `src/results.jl`: common `OptResult` output.
- `benchmark/benchmark.jl`: shared benchmark policy and orchestration only.

The relocation preserves solver defaults, arithmetic, branch order, counters and
termination conditions. It does not add ANCG stagnation or time-limit guards.
`Fast_NCG.jl` is the source filename; the exported constructor remains
`FastNCGSolver`. ANCG is loaded through the FastNCG package; no benchmark compatibility files are needed.

## Strongly convex local-convergence experiment

Run only FastNCG on the regularized power-chain problem:

```sh
julia --project=. experiments/strongly_convex.jl
```

The standalone script implements the objective, analytic gradient, HVP, and
stable objective difference. Defaults: d=100, phi=4, nu=0.01, 500 Gaussian
starts scaled by 0.1, seed=42, and gradient tolerance 1e-14. It records accepted
outer iterates without modifying the solver. Results (CSVs, summary, PNG and
PDF) are written to `experiments/results/strongly_convex/`.
Plotting uses Julia `Plots.jl` (GR backend). To compute data only or customize:

```julia
include("experiments/strongly_convex.jl")
results = StronglyConvexExperiment.run_experiment(; trials=500, plot=false)
```

The fitted slope is computed from the actual data; it is not fixed to the
manuscript's 2.070. Local order estimates are included in `error_pairs.csv`.

To redraw the saved convergence figure in Julia without rerunning the 500 trials:

```julia
include("experiments/strongly_convex.jl")
fig = StronglyConvexExperiment.plot_results()
display(fig)
```
