using LinearAlgebra
using Random
using Printf
using NLPModels
using FastNCG

# Baseline refreshed on 2026-10-10, Julia 1.12.6 / OpenBLAS ILP64, one thread.
# Deliberately tests the objective/gradient/HVP-only interface (no objdiff_fn).
# Keep exact oracle counts and the existing numerical comparison tolerances.
const REFERENCES = Dict(
    "Rosenbrock" => (
        iter=2517, hvp=15149, nobj=7567, ngrad=7567,
        f=1.4714557568952127e-22,
        g=4.1312684306971754e-10,
    ),
    "LogSumExp" => (
        iter=55, hvp=425, nobj=196, ngrad=196,
        f=1.4623357840219477,
        g=1.5020883855925921e-8,
    ),
    "Polytope" => (
        iter=47, hvp=79, nobj=89, ngrad=89,
        f=1.4004233017722169e-9,
        g=9.8264333066183967e-8,
    ),
    "TRegression" => (
        iter=173, hvp=1201, nobj=646, ngrad=646,
        f=1416.0209426997492,
        g=7.4287706408049945e-8,
    ),
)

function build_problems()
    problems = NamedTuple[]

    x0, o, g, hp = make_rosenbrock(1000; x0=zeros(1000))
    push!(problems, (name="Rosenbrock", x0=x0, obj=o, grad=g, hprod=hp))

    x0, o, g, hp = make_logsumexp(200, 1000, 0.05; x0=zeros(200))
    push!(problems, (name="LogSumExp", x0=x0, obj=o, grad=g, hprod=hp))

    x0, o, g, hp = make_polytope(400, 500, 4; x0=zeros(400))
    push!(problems, (name="Polytope", x0=x0, obj=o, grad=g, hprod=hp))

    x0, o, g, hp = make_tregression(200, 500, 0.001)
    push!(problems, (name="TRegression", x0=x0, obj=o, grad=g, hprod=hp))

    return problems
end

function run_fastncg(problem)
    nlp = FastNCG.ClosureNLPModel(
        copy(problem.x0), problem.obj, problem.grad, problem.hprod;
        name=problem.name,
    )

    solver = FastNCG.FastNCGSolver(max_outer=10000)
    reset!(nlp)

    elapsed = @elapsed x = FastNCG.optimize!(
        solver, nlp;
        initial_theta=copy(problem.x0),
        max_iter=10000,
        tol=1e-7,
    )

    # Evaluate final values only after recording algorithm counters.
    nobj = nlp.counters.neval_obj
    ngrad = nlp.counters.neval_grad
    f_final = obj(nlp, x)
    g_final = norm(grad(nlp, x))

    return (
        time=elapsed,
        status=solver.status,
        finite_iterates=all(isfinite,x) && all(isfinite,solver.loss_history) &&
                        all(isfinite,solver.grad_norm_history),
        decreasing_loss=all(diff(solver.loss_history) .<= 0),
        f=f_final,
        g=g_final,
        iter=length(solver.loss_history) - 1,
        hvp=solver.hvp_count,
        nobj=nobj,
        ngrad=ngrad,
    )
end

same_float(a, b; rtol=5e-5, atol=1e-14) =
    isapprox(a, b; rtol=rtol, atol=atol)

function regression_pass(r, ref)
    counts_ok = r.iter == ref.iter && r.hvp == ref.hvp &&
                r.nobj == ref.nobj && r.ngrad == ref.ngrad
    values_ok = same_float(r.f, ref.f) && same_float(r.g, ref.g)
    solution_ok = r.status == :first_order && isfinite(r.f) &&
                  isfinite(r.g) && 0 <= r.g <= 1e-7 &&
                  r.finite_iterates && r.decreasing_loss
    return counts_ok && values_ok && solution_ok
end

function _run_regression()
    println("="^110)
    println("FastNCG regression: current baseline (2026-10-10), one BLAS thread")
    println("="^110)
    @printf("%-14s %9s %9s %9s %9s %13s %13s %10s %10s\n",
            "Problem", "Iter", "HVP", "Obj", "Grad", "f(x*)", "|g(x*)|", "Time(s)", "Status")
    println("-"^110)

    all_pass = true

    for p in build_problems()
        r = run_fastncg(p)
        ref = REFERENCES[p.name]

        pass = regression_pass(r, ref)
        all_pass &= pass

        @printf("%-14s %9d %9d %9d %9d %13.4e %13.4e %10.4f %10s\n",
                p.name, r.iter, r.hvp, r.nobj, r.ngrad,
                r.f, r.g, r.time, pass ? "PASS" : "FAIL")

        if !pass
            println("  solver status=", r.status, ", finite iterates=", r.finite_iterates,
                    ", decreasing loss=", r.decreasing_loss)
            @printf("  reference: Iter=%d HVP=%d Obj=%d Grad=%d f=%.4e |g|=%.4e\n",
                    ref.iter, ref.hvp, ref.nobj, ref.ngrad, ref.f, ref.g)
        end
    end

    println("-"^110)
    if all_pass
        println("PASS.")
    else
        println("FAIL: reference mismatch or invalid/nonconverged solution.")
        error("FastNCG regression check failed.")
    end
    println("="^110)
end

function main()
    previous_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        _run_regression()
    finally
        BLAS.set_num_threads(previous_threads)
    end
end

main()
