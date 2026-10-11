# FastNCG.jl

**FastNCG.jl** is a Julia implementation of **FastNCG**, a matrix-free
second-order method for large-scale smooth optimization.

FastNCG computes regularized Newton steps using Hessian-vector products (HVPs)
and adaptive Krylov subspace approximations. It does not require explicitly
forming or factorizing the Hessian.

The package is built on
[`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
and also provides a lightweight `ClosureNLPModel` interface for user-defined
objectives.

## Installation

Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/FastNCG.git")
```

Then load the package:

```julia
using FastNCG
```

For development:

```bash
git clone https://github.com/Xunjian-Li/FastNCG.git
cd FastNCG
julia --project=.
```

and instantiate the environment:

```julia
using Pkg
Pkg.instantiate()
```

## Quick Start

FastNCG requires an objective, gradient, and Hessian-vector product.

For example, consider

$begin:math:display$
f\(x\)\=\\frac12 x\^\\top A x\-b\^\\top x\.
$end:math:display$

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

obj_fn(x) = 0.5 * dot(x, A * x) - dot(b, x)

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

solver = FastNCGSolver()

x = optimize!(
    solver,
    nlp;
    initial_theta=copy(x0),
    tol=1e-7,
    max_iter=1000,
)

println("Final objective     = ", solver.loss_history[end])
println("Final gradient norm = ", solver.grad_norm_history[end])
println("HVP evaluations     = ", solver.hvp_count)
```

The exact solution in this example is

```julia
x_exact = A \ b
```

so the numerical accuracy can be checked with

```julia
norm(x - x_exact) / norm(x_exact)
```

## Interface

A solver is created with

```julia
solver = FastNCGSolver()
```

and optimization is performed with

```julia
x = optimize!(
    solver,
    nlp;
    initial_theta=x0,
    tol=1e-7,
    max_iter=1000,
)
```

The main solver options are

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

After optimization, useful statistics are available through

```julia
solver.status
solver.hvp_count
solver.grad_count
solver.func_count
solver.krylov_extensions
solver.loss_history
solver.grad_norm_history
solver.time_history
solver.total_time
```

## Custom Problems

`ClosureNLPModel` provides a simple interface for custom objectives:

```julia
nlp = ClosureNLPModel(
    x0,
    obj_fn,
    grad_fn!,
    hprod_fn!;
    name="MyProblem",
)
```

where

```julia
obj_fn(x)
grad_fn!(g, x)
hprod_fn!(Hv, x, v)
```

compute respectively

$begin:math:display$
f\(x\)\, \\qquad \\nabla f\(x\)\, \\qquad \\nabla\^2 f\(x\)v\.
$end:math:display$

For large-scale problems, `hprod_fn!` should compute the HVP directly rather
than explicitly constructing the Hessian.

## Tests

Run the test suite with

```bash
julia --project=. test/runtests.jl
```

Additional numerical regression tests are available in

```bash
julia --project=. test/test.jl
```

The included test problems are:

- Rosenbrock
- Log-Sum-Exp
- Polytope
- Student $begin:math:text$t$end:math:text$ regression

## Benchmark

The benchmark compares FastNCG with several first- and second-order methods,
including ARCqK, ANCG, ARNCG, gradient descent, and Nesterov accelerated
gradient.

Run:

```bash
julia --project=. benchmark/benchmark.jl
```

The benchmark reports wall-clock time, iterations, HVP evaluations, objective
and gradient evaluations, final objective value, and gradient norm.

## Experiments

The strongly convex local-convergence experiment can be reproduced with

```bash
julia --project=. experiments/strongly_convex.jl
```

Results are written to

```text
experiments/results/strongly_convex/
```

## Citation

If you use FastNCG.jl in academic work, please cite the accompanying FastNCG
paper. Citation information will be added when the manuscript is publicly
available.

## Author

**Xun-Jian Li**  
Department of Biostatistics  
University of California, Los Angeles

## License

FastNCG.jl is released under the MIT License.