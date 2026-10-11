# ============================================================================
# benchmark_all_solvers.jl
#
# Formal benchmark using:
#   ARCqK, FastNCG, GD, NAG, CR, ANCG, ARNCG
#
# Design:
#   * Uses the validated current problem definitions from the FastNCG regression.
#   * Loads all algorithm implementations from the FastNCG package.
#   * Uses the installed FastNCG package; no embedded FastNCG implementation.
#   * One deterministic solve per method/problem, then BenchmarkTools sampling.
#   * Fresh deterministic problem/caches for every run.
#   * Reports median runtime and IQR, plus deterministic algorithmic counters.
# ============================================================================

using AdaptiveRegularization
using NLPModels
using LinearAlgebra
using Random
using Printf
using Statistics
using BenchmarkTools
using FastNCG

import NLPModels: reset!, obj, grad


# ============================================================================
# Benchmark wrappers
# ============================================================================

const BENCH_TOL = 1e-7
const BENCH_MAX_ITER = 10_000
const BENCH_SECONDS = 5.0
const ARC_MAX_TIME = 120.0

function make_problem(name::String)
    if name == "Rosenbrock"
        x0, o, g, hp, d = make_rosenbrock(1000; x0=zeros(1000))
        label = "Rosenbrock (dim=1000)"
    elseif name == "LogSumExp"
        x0, o, g, hp, d = make_logsumexp(200, 1000, 0.05; x0=zeros(200))
        label = "Log-Sum-Exp (dim=200, n=1000, rho=0.05)"
    elseif name == "Polytope"
        x0, o, g, hp, d = make_polytope(400, 500, 4; x0=zeros(400))
        label = "Polytope (dim=400, m=500, p=4)"
    elseif name == "TRegression"
        x0, o, g, hp, d = make_tregression(200, 500, 0.001)
        label = "T-Regression (dim=200, m=500, nu=0.001)"
    else
        error("Unknown problem: $name")
    end

    nlp = FastNCG.ClosureNLPModel(
        copy(x0), o, g, hp;
        objdiff_fn=d,
        name=name
    )

    return label, nlp, copy(x0)
end

function benchmark_arcqk(nlp; max_time=ARC_MAX_TIME, tol=BENCH_TOL)
    return arcqk_optimize(nlp; max_time=max_time, tol=tol)
end

function benchmark_fastncg(nlp, x0; tol=BENCH_TOL, max_outer=BENCH_MAX_ITER)
    solver = FastNCG.FastNCGSolver(max_outer=max_outer)

    t = @elapsed begin
        x_sol = FastNCG.optimize!(
            solver, nlp;
            initial_theta=copy(x0),
            max_iter=max_outer,
            tol=tol
        )
    end

    f_sol = obj(nlp, x_sol)
    g_sol = norm(grad(nlp, x_sol))

    return OptResult(
        "FastNCG", t, f_sol, g_sol,
        length(solver.loss_history)-1,
        solver.hvp_count,
        solver.func_count,
        solver.grad_count,
        copy(x_sol)
    )
end

function run_method(method::String, problem_name::String)
    _, nlp, x0 = make_problem(problem_name)

    if method == "ARCqK"
        return benchmark_arcqk(nlp; tol=BENCH_TOL)
    elseif method == "FastNCG"
        return benchmark_fastncg(nlp, x0; tol=BENCH_TOL, max_outer=BENCH_MAX_ITER)
    elseif method == "GD"
        return gd_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "NAG"
        return nag_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "CR"
        return cr_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "ANCG"
        return ancg_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "ARNCG"
        return arncg_official_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
            cg_maxiter=length(x0)+2,
        )
    else
        error("Unknown method: $method")
    end
end

function deterministic_run(method::String, problem_name::String)
    return run_method(method, problem_name)
end

function benchmark_method(method::String, problem_name::String; seconds=BENCH_SECONDS)
    # Fresh deterministic problem and mutable caches are constructed in setup.
    # BenchmarkTools excludes setup time from the measured expression.
    #
    # Do not interpolate `seconds` with `$seconds` here. BenchmarkTools parses
    # keyword parameters such as `seconds` outside the benchmark expression.
    params = BenchmarkTools.Parameters(seconds=seconds, evals=1)
    trial = @benchmarkable begin
        run_method_on_instance($method, nlp, x0)
    end setup=begin
        problem_label, nlp, x0 = make_problem($problem_name)
    end
    trial.params = params
    result = run(trial)

    med_ns = median(result).time
    q25_ns = quantile(result.times, 0.25)
    q75_ns = quantile(result.times, 0.75)

    return (
        trial=result,
        median=med_ns / 1e9,
        q25=q25_ns / 1e9,
        q75=q75_ns / 1e9,
        iqr=(q75_ns - q25_ns) / 1e9,
        samples=length(result.times),
    )
end

function run_method_on_instance(method::String, nlp, x0)
    if method == "ARCqK"
        return benchmark_arcqk(nlp; tol=BENCH_TOL)
    elseif method == "FastNCG"
        return benchmark_fastncg(nlp, x0; tol=BENCH_TOL, max_outer=BENCH_MAX_ITER)
    elseif method == "GD"
        return gd_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "NAG"
        return nag_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "CR"
        return cr_optimize(
            nlp;
            initial_theta=x0,
            max_iter=BENCH_MAX_ITER,
            tol=BENCH_TOL,
        )
    elseif method == "ANCG"
        return ancg_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "ARNCG"
        return arncg_official_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
            cg_maxiter=length(x0)+2,
        )
    else
        error("Unknown method: $method")
    end
end

function print_summary(rows)
    println()
    println("="^178)
    println("Formal benchmark summary (BenchmarkTools, evals=1, seconds=$(BENCH_SECONDS) per method/problem)")
    println("Timing excludes deterministic problem/cache construction in setup.")
    println("Algorithmic counters and final values come from one separate deterministic run.")
    println("ANCG uses explicit Hessians: HVP includes Hessian reconstruction plus CG matrix-vector products.")
    println("="^178)

    @printf(
        "%-13s %-9s %11s %11s %11s %8s %9s %8s %8s %8s %13s %12s %-14s\n",
        "Problem", "Method", "Median(s)", "Q25(s)", "Q75(s)", "Samples",
        "Iter", "HVP", "Obj", "Grad", "f(x*)", "|g(x*)|", "Status"
    )
    println("-"^178)

    current_problem = ""
    for r in rows
        if current_problem != "" && r.problem != current_problem
            println("-"^178)
        end
        current_problem = r.problem

        @printf(
            "%-13s %-9s %11.5f %11.5f %11.5f %8d %9d %8d %8d %8d %13.5e %12.4e %-14s\n",
            r.problem, r.method, r.median, r.q25, r.q75, r.samples,
            r.iter, r.hvp, r.nobj, r.ngrad, r.f, r.g, r.status
        )
    end
    println("="^178)
end

function main()
    problem_names = ["Rosenbrock", "LogSumExp", "Polytope", "TRegression"]
    methods = ["ARCqK", "FastNCG", "GD", "NAG", "CR", "ANCG", "ARNCG"]

    println("="^108)
    println("Formal optimizer benchmark")
    println("Methods: ", join(methods, ", "))
    println("BenchmarkTools protocol: seconds=$(BENCH_SECONDS), evals=1")
    println("Tolerance: $(BENCH_TOL)")
    println("="^108)

    rows = NamedTuple[]

    for problem_name in problem_names
        label, _, _ = make_problem(problem_name)

        println()
        println("="^108)
        println("Problem: ", label)
        println("="^108)

        for method in methods
            # Run once outside BenchmarkTools to record deterministic algorithmic metrics.
            print(@sprintf("  %-8s deterministic run ... ", method))
            r = deterministic_run(method, problem_name)
            status = r.g_final <= BENCH_TOL ? "CONVERGED" : "NOT CONVERGED"
            @printf(
                "iter=%d HVP=%d f=%.5e |g|=%.4e [%s]\n",
                r.iter, r.hvp, r.f_final, r.g_final, status
            )

            if method == "ANCG"
                @printf("           CG-HVP=%d Hessian-build-HVP=%d Hessian-evals=%d; %s\n",
                        r.details.hvp_count, r.details.hessian_hvp,
                        r.details.Hevals, r.details.stop)
            end

            # BenchmarkTools performs compilation warm-up and repeated sampling.
            print(@sprintf("  %-8s benchmark ... ", method))
            b = benchmark_method(method, problem_name)
            @printf(
                "median=%.5f s  Q25=%.5f  Q75=%.5f  samples=%d\n",
                b.median, b.q25, b.q75, b.samples
            )

            push!(rows, (
                problem=problem_name,
                method=method,
                median=b.median,
                q25=b.q25,
                q75=b.q75,
                iqr=b.iqr,
                samples=b.samples,
                iter=r.iter,
                hvp=r.hvp,
                nobj=r.nobj,
                ngrad=r.ngrad,
                f=r.f_final,
                g=r.g_final,
                status=status,
                details=r.details,
            ))
        end
    end

    print_summary(rows)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = main()
end
