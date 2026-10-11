using Test
using LinearAlgebra
using Random
using FastNCG

objective_change(diff, x, y) = first(diff(x, y))

@testset "Problem callbacks and cache invalidation" begin
    factories = [
        () -> make_rosenbrock(8),
        () -> make_logsumexp(8, 30, 0.5),
        [() -> make_polytope(8, 30, p) for p in 2:5]...,
        () -> make_tregression(8, 30, 0.5),
    ]
    rng = MersenneTwister(123)
    for factory in factories
        x, obj, grad!, hp!, diff = factory()
        _, fresh_obj, fresh_grad!, fresh_hp!, fresh_diff = factory()
        v = normalize(randn(rng, length(x)))
        g, h, expected = similar(x), similar(x), similar(x)
        trial = x + 0.02v

        # Gradient first must not leave a stale cached objective.
        grad!(g, x)
        @test obj(x) ≈ fresh_obj(x)
        @test obj(copy(x)) == obj(x)
        hp!(h, x, v)
        @test h ≈ fresh_hp!(expected, x, v)
        for scale in (1.0, 2.0, -0.5)
            trial .= x .+ (0.02scale) .* v  # Same mutable trial buffer.
            @test obj(trial) ≈ fresh_obj(trial)
            @test grad!(g, trial) ≈ fresh_grad!(expected, trial)
            @test diff(x, trial)[1] ≈ (fresh_obj(trial) - fresh_obj(x)) atol=1e-10 rtol=1e-9
            # Trial evaluations must not overwrite the outer Hessian cache.
            @test hp!(h, x, v) ≈ fresh_hp!(expected, x, v)
        end
        @test obj(x) ≈ fresh_obj(x)  # Return to cached Hessian point.
        x .+= 0.01v                 # Mutate the outer buffer as FastNCG does.
        @test grad!(g, x) ≈ fresh_grad!(expected, x)
        @test obj(x) ≈ fresh_obj(x)
        @test hp!(h, x, v) ≈ fresh_hp!(expected, x, v)
        @test diff(x, trial)[1] ≈ fresh_diff(x, trial)[1] atol=1e-10 rtol=1e-9

        # Independent finite differences for derivatives, away from kinks.
        eps_fd = 1e-6
        grad!(g, x)
        @test dot(g, v) ≈ (fresh_obj(x + eps_fd*v) - fresh_obj(x - eps_fd*v))/(2eps_fd) rtol=1e-5 atol=1e-6
        gp, gm = similar(x), similar(x)
        fresh_grad!(gp, x + eps_fd*v)
        fresh_grad!(gm, x - eps_fd*v)
        @test hp!(h, x, v) ≈ (gp - gm)/(2eps_fd) rtol=1e-5 atol=1e-6

        # Tiny-step differences must stay consistent with the directional derivative.
        tiny_trial = x + 1e-8v
        obj(tiny_trial)
        @test (@inferred objective_change(diff, x, tiny_trial)) isa Float64
        @test diff(x, tiny_trial)[1] ≈ 1e-8dot(g, v) rtol=1e-4 atol=1e-11
    end
end

@testset "FastNCG with stable objective differences" begin
    for problem in (make_rosenbrock(8), make_logsumexp(8, 30, 0.5),
                    make_polytope(8, 30, 3), make_tregression(8, 30, 0.5))
        x0, obj, grad!, hp!, diff = problem
        model = FastNCG.ClosureNLPModel(x0, obj, grad!, hp!; objdiff_fn=diff)
        solver = FastNCG.FastNCGSolver(max_outer=1000)
        x = FastNCG.optimize!(solver, model; initial_theta=x0, tol=1e-6)
        g = similar(x)
        @test norm(grad!(g, x)) <= 1e-6
        @test obj(x) <= obj(x0)
        @test solver.hvp_count > 0
    end
end

function quartic_model(offset, coefficient; objdiff_fn=nothing)
    f(x) = offset + (-x[1] + coefficient * x[1]^4)
    g!(g, x) = (g[1] = -1 + 4coefficient * x[1]^3; g)
    hp!(h, x, v) = (h[1] = 12coefficient * x[1]^2 * v[1]; h)
    return ClosureNLPModel([0.0], f, g!, hp!; objdiff_fn=objdiff_fn)
end

@testset "Actual descent and independent regularization retries" begin
    # A large additive constant must not allow the quadratic model to
    # override an observable objective increase, or a rounded-to-zero change.
    for diff in (nothing, (x, y) -> (NaN, Inf))
        model = quartic_model(1e16, 100.0; objdiff_fn=diff)
        solver = FastNCGSolver(max_regularization_retries=12)
        x = optimize!(solver, model; initial_theta=[0.0], max_iter=1)
        @test x == [0.0]
        @test solver.status == :inner_failure
        @test solver.ratio_increases == 12
        @test length(solver.loss_history) == 1
    end

    # A stable difference can still certify small decreases under that offset.
    diff(x, y) = ((x[1] - y[1]) + 100(y[1]^4 - x[1]^4), nothing)
    model = quartic_model(1e16, 100.0; objdiff_fn=diff)
    solver = FastNCGSolver()
    x = optimize!(solver, model; initial_theta=[0.0], max_iter=1)
    @test diff([0.0], x)[1] < 0
    @test solver.status == :max_iter

    model = quartic_model(0.0, 1000.0)
    solver = FastNCGSolver(max_regularization_retries=0)
    @test optimize!(solver, model; initial_theta=[0.0], max_iter=1) == [0.0]
    @test solver.ratio_increases == 0
    @test solver.status == :inner_failure

    solver = FastNCGSolver(max_regularization_retries=50, max_krylov=1)
    x = optimize!(solver, model; initial_theta=[0.0], max_iter=1)
    @test model.obj_fn(x) < model.obj_fn([0.0])
    @test solver.ratio_increases > 3
    @test solver.status == :max_iter
    x = optimize!(solver, model; initial_theta=[0.0], max_iter=100, tol=1e-6)
    @test solver.status == :first_order
    @test norm(model.grad_fn(zeros(1), x)) <= 1e-6
    # Status and counters must reset when reusing a solver.
    optimize!(solver, model; initial_theta=[0.0], max_iter=0)
    @test solver.status == :max_iter
    @test solver.ratio_increases == 0
end

@testset "Reused tridiagonal factors and growing Krylov storage" begin
    rng = MersenneTwister(912)
    for n in (1, 2, 17, 64)
        off = randn(rng, n - 1)
        diag = fill(2.0, n)
        for i in eachindex(off)
            diag[i] += abs(off[i])
            diag[i+1] += abs(off[i])
        end
        T = SymTridiagonal(diag, off)
        b = randn(rng, n)
        u, w, cp, dp = [zeros(n) for _ in 1:4]
        pivots = copy(diag)
        FastNCG._solve_tridiagonal_ws!(u, pivots, off, b, cp, dp, pivots)
        @test u ≈ T \ b
        FastNCG._solve_factored_tridiagonal_ws!(w, pivots, off, u, cp, dp)
        @test w ≈ T \ u
        @test FastNCG._min_eig_tridiag(diag, off) ≈ minimum(eigvals(T))
    end

    # Growing must preserve all computed basis vectors and the next vector.
    Q = randn(rng, 40, 17)
    original = copy(Q)
    @test FastNCG._ensure_krylov_capacity(Q, 17, 41) === Q
    Q = FastNCG._ensure_krylov_capacity(Q, 18, 41)
    @test size(Q) == (40, 34)
    @test Q[:, 1:17] == original
    Q[:, 18:34] .= randn(rng, 40, 17)
    original = copy(Q)
    Q = FastNCG._ensure_krylov_capacity(Q, 35, 41)
    @test size(Q) == (40, 41)
    @test Q[:, 1:34] == original
end

@testset "Lazy objective differences and initial point" begin
    forbidden_diff(x, y) = error("difference evaluated for an ineligible trial")
    # Gradient condition fails on the initial quartic trial.
    model = quartic_model(0.0, 1e6; objdiff_fn=forbidden_diff)
    solver = FastNCGSolver(max_regularization_retries=0)
    optimize!(solver, model; max_iter=1)
    @test solver.status == :inner_failure
    @test solver.func_count == 2

    # Projection fails, while the gradient condition passes.
    H = [1.0 10.0; 10.0 100.0]
    f(x) = dot(x, H*x)/2 + x[1]
    g!(g, x) = (mul!(g, H, x); g[1] += 1; g)
    hp!(h, x, v) = mul!(h, H, v)
    model = ClosureNLPModel(zeros(2), f, g!, hp!; objdiff_fn=forbidden_diff)
    solver = FastNCGSolver(tau=0.5, max_krylov=1, max_regularization_retries=0)
    optimize!(solver, model; max_iter=1)
    @test solver.status == :inner_failure
    @test solver.func_count == 2

    calls = Ref(0)
    diff(x, y) = (calls[] += 1; ((y[1]-x[1])*(y[1]+x[1]-2)/2, nothing))
    model = ClosureNLPModel([2.0], x -> (x[1]-1)^2/2,
        (g, x) -> (g[1] = x[1]-1; g), (h, x, v) -> copyto!(h, v);
        objdiff_fn=diff)
    solver = FastNCGSolver()
    initial = optimize!(solver, model; max_iter=0)
    @test initial == [2.0]
    @test initial !== model.meta.x0
    x = optimize!(solver, model; max_iter=1)
    @test 1.0 < x[1] < 2.0
    @test calls[] == 1
    @test model.meta.x0 == [2.0]
    explicit = [3.0]
    x = optimize!(solver, model; initial_theta=explicit, max_iter=1)
    @test 1.0 < x[1] < 3.0
    @test explicit == [3.0]
    @test model.meta.x0 == [2.0]
end

@testset "Floating-point stagnation" begin
    model = ClosureNLPModel([1e20], x -> x[1],
        (g, x) -> fill!(g, 1.0), (h, x, v) -> fill!(h, 0.0);
        objdiff_fn=(x, y) -> error("unchanged point must not be evaluated"))
    solver = FastNCGSolver()
    x = optimize!(solver, model; max_iter=10)
    @test x == model.meta.x0
    @test solver.status == :numerical_stagnation
    @test solver.func_count == 1
    @test solver.grad_count == 1
    @test solver.ratio_increases == 0
    @test length(solver.loss_history) == 1
    @test last(solver.grad_norm_history) == 1.0
    optimize!(solver, model; initial_theta=[0.0], max_iter=0)
    @test solver.status == :max_iter

    # An already stationary point reports convergence, not stagnation.
    stationary = ClosureNLPModel([1e20], x -> 0.0,
        (g, x) -> fill!(g, 0.0), (h, x, v) -> fill!(h, 0.0))
    optimize!(solver, stationary)
    @test solver.status == :first_order
end

include("ancg_reference.jl")
