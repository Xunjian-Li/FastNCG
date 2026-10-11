# Paper reproduction settings from AdaptiveRegularization 0.2.2,
# paper/benchmark_arctr.jl. Benchmark-wide stopping tolerances remain shared.
function arcqk_paper_options()
    TR = AdaptiveRegularization.ARTrustRegion(10.0;
        acceptance_threshold=0.1, increase_threshold=0.75,
        increase_factor=5.0, decrease_factor=0.1)
    return (TR=TR, shifts=10.0 .^ collect(-10.0:0.5:20.0), ζ=0.5)
end

function arcqk_optimize(nlp; max_time=120.0, tol=1e-7)
    reset!(nlp)
    t = @elapsed begin
        stats = ARCqKOp(
            nlp;
            arcqk_paper_options()...,
            max_time=max_time,
            max_iter=typemax(Int64),
            max_eval=typemax(Int64),
            atol=tol,
            rtol=0.0,
        )
    end

    return OptResult(
        "ARCqK", t, stats.objective, stats.dual_feas, stats.iter,
        neval_hprod(nlp), neval_obj(nlp), neval_grad(nlp),
        copy(stats.solution),
    )
end
