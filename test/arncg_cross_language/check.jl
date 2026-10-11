using Test
include(joinpath(@__DIR__, "..", "..", "benchmark", "benchmark.jl"))
BLAS.set_num_threads(1)
root = ARGS[1]
mkpath(root)
function fixture(name)
    if name == "stationary" || name == "quadratic"
        x0 = name == "stationary" ? [0.0,0.0] : [1.0,-1.0]
        H = [4.0 1.0; 1.0 2.0]
        return ClosureNLPModel(x0, x->dot(x,H*x)/2,
            (g,x)->mul!(g,H,x), (h,x,v)->mul!(h,H,v))
    elseif name == "rosenbrock"
        x0,o,g,h,d = make_rosenbrock(2; x0=[-1.2,1.0])
        return ClosureNLPModel(x0,o,g,h)
    else
        return ClosureNLPModel([0.1,1.0], x->(x[1]^2-1)^2/4+x[2]^2/2,
            (g,x)->(g .= [x[1]^3-x[1],x[2]]),
            (h,x,v)->(h .= [(3*x[1]^2-1)*v[1],v[2]]))
    end
end
@testset "ARNCG reference and ARCqK paper configuration" begin
    for name in ("stationary","quadratic","rosenbrock","negative_curvature")
        p = fixture(name)
        r = arncg_official_optimize(p;initial_theta=p.meta.x0,tol=1e-7)
        @test r.g_final < 1e-7
        @test r.iter == length(r.details.fx_hist)
        @test r.hvp == p.counters.neval_hprod
        if name == "stationary"
            @test r.iter == r.hvp == 0
        end
        if name == "negative_curvature"
            @test :NC in r.details.dtype_hist
        end
        open(joinpath(root,name*"_julia.csv"),"w") do io
            println(io,join([r.iter,r.nobj,r.ngrad,r.hvp,r.f_final,r.g_final,r.x...],","))
        end
        open(joinpath(root,name*"_julia_history.csv"),"w") do io
            for k in 1:r.iter
                d=r.details
                println(io,join([d.fx_hist[k],d.gn_hist[k],d.alpha_hist[k],d.M_hist[k],d.cg_hist[k],d.dtype_hist[k]==:SOL ? 1 : 2],","))
            end
        end
        println(name, " ", (iter=r.iter,hvp=r.hvp,loss=r.f_final,gnorm=r.g_final))
    end
    p=fixture("quadratic")
    a=benchmark_arcqk(p;tol=1e-7)
    @test a.g_final < 1e-7
    @test a.hvp == p.counters.neval_hprod
end
