using Test
include(joinpath(@__DIR__, "..", "benchmark", "benchmark.jl"))

@testset "ANCG benchmark adapter" begin
    H = [4.0 1.0; 1.0 2.0]
    x0 = [1.0, -1.0]
    f(x) = dot(x, H*x)/2
    g!(g, x) = mul!(g, H, x)
    hp!(h, x, v) = mul!(h, H, v)
    model = ClosureNLPModel(x0, f, g!, hp!)
    a = ancg_optimize(model; initial_theta=x0, tol=1e-7)
    @test a.details.hessian_hvp == length(x0)*a.details.Hevals
    @test a.hvp == a.details.hvp_count + a.details.hessian_hvp
    @test a.nobj == model.counters.neval_obj == a.details.fevals
    @test a.ngrad == model.counters.neval_grad == a.details.gevals
    @test model.counters.neval_hprod == a.details.hessian_hvp
    @test a.g_final <= 1e-7
    @test a.details.stop == "First-order tolerance satisfied."
    b = ancg_optimize(model; initial_theta=x0, tol=1e-7, hess_fn=x -> H)
    @test a.x == b.x
    @test a.details.fx_hist == b.details.fx_hist
    @test a.details.gn_hist == b.details.gn_hist
    @test a.details.dtype_hist == b.details.dtype_hist
    @test a.details.hvp_count == b.details.hvp_count
    @test b.details.hessian_hvp == 0
    @test x0 == [1.0, -1.0]
    # Existing optimizers still construct the original nine-argument result.
    old = OptResult("test", 0.0, 0.0, 0.0, 0, 0, 0, 0, [0.0])
    @test isempty(old.details)
end
