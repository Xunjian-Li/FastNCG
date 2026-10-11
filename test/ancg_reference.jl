using Test
using LinearAlgebra
using FastNCG: ANCGReference

@testset "Authors' capCG reference" begin
    d, kind, dHd, calls, it = ANCGReference.capcg(reshape([2.0], 1, 1), [1.0], 1.0, 0.1)
    @test d == [-0.25]
    @test kind == 1
    @test dHd == 0.125
    @test (calls, it) == (5, 1)

    # Preserve the source's missing early return: p0 is -1, yet the loop
    # proceeds and returns yj = +0.5 after all five counted products.
    d, kind, dHd, calls, it = ANCGReference.capcg(reshape([-4.0], 1, 1), [1.0], 1.0, 0.1)
    @test d == [0.5]
    @test kind == 2
    @test dHd == -1.0
    @test (calls, it) == (5, 1)
    @test_throws UndefVarError ANCGReference.capcg(reshape([2.0], 1, 1), [1.0], 1.0, 0.1, [0.0])

    H = [4.0 1.0; 1.0 2.0]
    g = [1.0, -2.0]
    d, kind, dHd, calls, it = ANCGReference.capcg(H, g, 0.1, 0.1)
    @test kind == 1
    @test norm((H + 0.2I)*d + g) < 1e-12
    @test dHd ≈ dot(d, H*d)
    @test calls == 2 + 3it
end

@testset "Authors' ANCG outer flow" begin
    H(x) = reshape([1.0], 1, 1)
    f(x) = x[1]^2/2
    g(x) = copy(x)
    x0 = [1.0]
    x, out = ANCGReference.optimize(f, g, H, x0; maxit=1, verbose=false)
    @test x0 == [1.0]
    @test x ≈ [2/3]
    # Full trial does not halve the gradient, so it is evaluated again by
    # Armijo; accepted f/g/H are reused at the next outer iteration.
    @test (out.iters, out.fevals, out.gevals, out.Hevals, out.hvp_count) == (1, 3, 3, 2, 5)
    @test out.alpha_hist == [1.0]
    @test out.dtype_hist == [1]
    @test out.gamma_hist == [1.0, 1.0]
    @test out.stop == "Max iterations reached."
    x, out = ANCGReference.optimize(f, g, H, x0; verbose=false)
    @test norm(x) <= 1e-6
    @test out.stop == "First-order tolerance satisfied."
    @test length(out.fx_hist) == out.iters + 1
    @test length(out.gn_hist) == out.iters + 1
    @test length(out.alpha_hist) == out.iters
    @test out.Hevals == out.iters + 1
    @test all(diff(out.fx_hist) .<= 0)

    _, out = ANCGReference.optimize(f, g, H, [0.0]; verbose=false)
    @test (out.iters, out.fevals, out.gevals, out.Hevals, out.hvp_count) == (0, 1, 1, 1, 0)
    @test out.stop == "First-order tolerance satisfied."

    # Both line-search failure branches stop immediately, including maxls=0
    # (one Armijo trial), without changing gamma or taking a fallback step.
    for (h0, expected) in ((0.0, "SOL"), (-4.0, "NC"))
        fbad(x) = x[1] + h0*x[1]^2/2 + 1000x[1]^4
        gbad(x) = [1 + h0*x[1] + 4000x[1]^3]
        Hbad(x) = reshape([h0 + 12000x[1]^2], 1, 1)
        x, out = ANCGReference.optimize(fbad, gbad, Hbad, [0.0]; maxls=0, verbose=false)
        @test x == [0.0]
        @test out.stop == "Line search failed ($expected branch)."
        @test out.iters == 1
        @test out.fevals == (expected == "SOL" ? 3 : 2)
        @test out.gevals == out.Hevals == 1
        @test out.gamma_hist == [1.0]
        @test isempty(out.alpha_hist)
    end
    # Inclusive j <= maxls: a second Armijo trial is allowed when maxls=1.
    fstep(x) = -x[1] + 10x[1]^4
    gstep(x) = [-1 + 40x[1]^3]
    Hstep(x) = reshape([120x[1]^2], 1, 1)
    x, out = ANCGReference.optimize(fstep, gstep, Hstep, [0.0]; maxls=1, maxit=1, verbose=false)
    @test out.alpha_hist == [0.5]
    @test x == [0.25]
    @test out.fevals == 4

    # Controlled callbacks exercise gamma growth/capping independently of
    # the differential properties of a particular optimization problem.
    ftiny(x) = 1e-6*x[1]
    gconstant(x) = [1.0]
    Hlarge(x) = reshape([1e8], 1, 1)
    _, out = ANCGReference.optimize(ftiny, gconstant, Hlarge, [0.0];
        maxit=3, gamma_cap=4.0, verbose=false)
    @test out.gamma_hist == [1.0, 2.0, 4.0, 4.0]
    _, out = ANCGReference.optimize(ftiny, gconstant, Hlarge, [0.0];
        maxit=21, gamma_cap=16.0, restart=true, verbose=false)
    @test out.gamma_hist[21] == 16.0
    @test out.gamma_hist[22] == 2.0  # Restart occurs after iteration 20 logging.
    _, out = ANCGReference.optimize(f, g, H, [0.0]; gamma0=0.1, verbose=false)
    @test out.gamma_hist == [1.0]

    for opts in ((eta=0.0,), (zeta=1.0,), (theta=1.0,))
        @test_throws ErrorException ANCGReference.optimize(f, g, H, x0; opts..., verbose=false)
    end
end
