"""
Standalone FastNCG local-convergence experiment.

Run from the project root:
    julia --project=. experiments/strongly_convex.jl
Or include this file and call:
    StronglyConvexExperiment.run_experiment(; trials=500, seed=42)

Only FastNCG is run. Plotting uses Julia Plots.jl with the GR backend.
Use plot=false to generate the numerical data only.
"""
module StronglyConvexExperiment

using FastNCG
using LinearAlgebra
using Random
using Printf
using Statistics
import Plots
using LaTeXStrings


"""
    make_strongly_convex(d=100; phi=4.0, nu=1e-2, x0=zeros(d))

Return `(x0, obj, grad!, hprod!, objdiff)` for
    sum(abs.(D*x).^phi)/phi + nu*dot(x,x)/2,
where `(D*x)[i]=x[i]-x[i+1]` for i<d and `(D*x)[d]=x[d]`.
The known minimizer is zero. Products use the bidiagonal D directly in O(d).
"""
function make_strongly_convex(d::Int=100; phi::Real=4.0, nu::Real=1e-2,
                              x0=zeros(d))
    d >= 2 || throw(ArgumentError("d must be at least 2"))
    isfinite(phi) && phi >= 2 || throw(ArgumentError("phi must be finite and at least 2"))
    isfinite(nu) && nu > 0 || throw(ArgumentError("nu must be finite and positive"))
    length(x0) == d || throw(DimensionMismatch("length(x0) must equal d"))
    p, reg = Float64(phi), Float64(nu)
    function objective(x)
        total = abs(x[d])^p
        @inbounds for i in 1:d-1
            total += abs(x[i]-x[i+1])^p
        end
        return total/p + reg*dot(x,x)/2
    end
    function gradient!(g,x)
        @inbounds for i in 1:d
            g[i] = reg*x[i]
        end
        @inbounds for i in 1:d-1
            r = x[i]-x[i+1]
            t = abs(r)^(p-2)*r
            g[i] += t
            g[i+1] -= t
        end
        g[d] += abs(x[d])^(p-2)*x[d]
        return g
    end
    function hessian_product!(h,x,v)
        @inbounds for i in 1:d
            h[i] = reg*v[i]
        end
        @inbounds for i in 1:d-1
            t = (p-1)*abs(x[i]-x[i+1])^(p-2)*(v[i]-v[i+1])
            h[i] += t
            h[i+1] -= t
        end
        h[d] += (p-1)*abs(x[d])^(p-2)*v[d]
        return h
    end
    function power_difference(a,b)
        # Exact factorization avoids subtracting close fourth powers.
        p == 4 && return (a-b)*(a+b)*(a*a+b*b)
        p == 2 && return (a-b)*(a+b)
        aa, bb = abs(a), abs(b)
        hi, lo = max(aa,bb), min(aa,bb)
        hi == 0 && return 0.0
        delta = hi^p * (-expm1(p*log1p((lo-hi)/hi)))
        return aa >= bb ? delta : -delta
    end
    function objective_difference(x,y)
        total = power_difference(y[d],x[d])/p
        @inbounds for i in 1:d-1
            total += power_difference(y[i]-y[i+1],x[i]-x[i+1])/p
        end
        @inbounds for i in 1:d
            total += (reg/2)*(y[i]-x[i])*(y[i]+x[i])
        end
        return total
    end
    return Vector{Float64}(x0), objective, gradient!, hessian_product!, objective_difference
end

function solve_trial(x0; phi=4.0, nu=1e-2, tol=1e-14, max_iter=500,
                     solver_kwargs=NamedTuple())
    x0, f, g!, hv!, diff = make_strongly_convex(length(x0); phi, nu, x0)
    errors = [norm(x0)]
    last_base = copy(x0)
    function record_base!(x)
        if x != last_base
            push!(errors,norm(x))
            copyto!(last_base,x)
        end
    end
    function observed_hvp!(h,x,v)
        # Current FastNCG evaluates HVPs only at the accepted outer base point,
        # not rejected trial points. Record each new base once; append the final
        # returned point below (it may stop before another HVP is requested).
        record_base!(x)
        return hv!(h,x,v)
    end
    model = ClosureNLPModel(x0,f,g!,observed_hvp!;
        objdiff_fn=diff,name="StronglyConvexChain")
    solver = FastNCGSolver(; max_outer=max_iter, solver_kwargs...)
    x = optimize!(solver,model; tol=Float64(tol),max_iter=max_iter)
    record_base!(x)
    length(errors) == length(solver.loss_history) ||
        error("Recorded base points no longer match accepted-iterate history")
    return (errors=errors,losses=copy(solver.loss_history),
        gradients=copy(solver.grad_norm_history), status=solver.status,
        iterations=length(errors)-1,hvp=solver.hvp_count,x=x)
end

function pooled_fit(pairs)
    length(pairs) >= 2 || return (slope=NaN,intercept=NaN,r2=NaN)
    xs = [p.log_error for p in pairs]
    ys = [p.log_next_error for p in pairs]
    xc, yc = xs .- mean(xs), ys .- mean(ys)
    denom = dot(xc,xc)
    denom > 0 || return (slope=NaN,intercept=NaN,r2=NaN)
    slope = dot(xc,yc)/denom
    intercept = mean(ys)-slope*mean(xs)
    residual = ys .- (intercept .+ slope.*xs)
    r2 = dot(yc,yc) == 0 ? NaN : 1-dot(residual,residual)/dot(yc,yc)
    return (;slope,intercept,r2)
end

"""
    plot_results(output_dir=joinpath(@__DIR__, "results", "strongly_convex"), fit=nothing)

Replot saved error_pairs.csv entirely in Julia, without rerunning the solver.
The optional fit is the result from pooled_fit; otherwise recompute it from CSV.
Save LocalConvergence.png and LocalConvergence.pdf, and return the Plots figure.
"""
function plot_results(output_dir=@__DIR__, fit=nothing)
    pairs = NamedTuple[]
    for line in Iterators.drop(eachline(joinpath(output_dir,"error_pairs.csv")),1)
        isempty(strip(line)) && continue
        fields = split(line,',')
        push!(pairs,(log_error=parse(Float64,fields[5]),
                     log_next_error=parse(Float64,fields[6])))
    end
    isempty(pairs) && throw(ArgumentError("No error pairs to plot"))

    fit = fit === nothing ? pooled_fit(pairs) : fit
    xs = [p.log_error for p in pairs]
    ys = [p.log_next_error for p in pairs]

    Plots.gr()

    fig = Plots.scatter(xs,ys;
        marker=:diamond,markersize=3.5,markeralpha=0.35,markerstrokewidth=0,
        color="#4C78A8",label="FastNCG",
        xlabel=L"\log \|x_k-x^\star\|_2",
        ylabel=L"\log \|x_{k+1}-x^\star\|_2",
        legend=:topleft,gridalpha=0.15,framestyle=:box,
        size=(600,450),dpi=240,legendfontsize=10,guidefontsize=13,
        tickfontsize=10,left_margin=10*Plots.mm,bottom_margin=6*Plots.mm)

    line_x = range(minimum(xs),maximum(xs);length=300)

    if isfinite(fit.slope)
        Plots.plot!(fig,line_x,fit.intercept .+ fit.slope.*line_x;
            color="#c23937",linestyle=:dash,linewidth=2,
            label=@sprintf("Empirical fit (slope = %.2f)",fit.slope))
    end

    reference_intercept = mean(ys .- 2 .* xs)
    Plots.plot!(fig,line_x,reference_intercept .+ 2 .* line_x;
        color=:black,linestyle=:dot,linewidth=2,
        label="Quadratic rate (slope = 2)")

    Plots.savefig(fig,joinpath(output_dir,"LocalConvergence.png"))
    Plots.savefig(fig,joinpath(output_dir,"LocalConvergence.pdf"))

    display(fig)
    return fig
end

"""
    run_experiment(; d=100, phi=4.0, nu=1e-2, trials=500, seed=42, ...)

Each starting point is `0.1*randn(rng,d)` (not normalized). CSV files retain all
runs, including failures. Natural-log pairs with zero/nonfinite errors are
excluded from regression and counted explicitly. No slope or convergence claim
is assumed; the pooled fit includes transient and terminal pairs as requested.
"""
function run_experiment(; d::Int=100, phi=4.0, nu=1e-2,
        trials::Int=500, seed::Int=42, initial_scale=0.1,
        tol=1e-14, max_iter::Int=500, solver_kwargs=NamedTuple(),
        output_dir=@__DIR__, plot::Bool=true)
    trials > 0 || throw(ArgumentError("trials must be positive"))
    max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))
    isfinite(initial_scale) && initial_scale > 0 || throw(ArgumentError("initial_scale must be positive and finite"))
    # Validate problem parameters before starting any runs.
    make_strongly_convex(d;phi,nu)
    rng = MersenneTwister(seed)
    runs = NamedTuple[]
    pairs = NamedTuple[]
    excluded = 0
    for trial in 1:trials
        r = solve_trial(initial_scale*randn(rng,d);phi,nu,tol,max_iter,solver_kwargs)
        push!(runs,r)
        for k in 1:length(r.errors)-1
            en, ep = r.errors[k], r.errors[k+1]
            if isfinite(en) && isfinite(ep) && en > 0 && ep > 0
                local_order = NaN
                if k >= 2 && r.errors[k-1] > 0 && en != r.errors[k-1]
                    local_order = log(ep/en)/log(en/r.errors[k-1])
                end
                push!(pairs,(trial=trial,iteration=k-1,error=en,next_error=ep,
                    log_error=log(en),log_next_error=log(ep),local_order=local_order))
            else
                excluded += 1
            end
        end
        if trial % 50 == 0 || trial == trials
            @printf("FastNCG: %d/%d runs, latest status=%s, |g|=%.3e\n",
                trial,trials,string(r.status),last(r.gradients))
            flush(stdout)
        end
    end
    fit = pooled_fit(pairs)
    output_dir = abspath(output_dir)
    mkpath(output_dir)
    open(joinpath(output_dir,"error_pairs.csv"),"w") do io
        println(io,"trial,iteration,error,next_error,log_error,log_next_error,local_order")
        for p in pairs
            println(io,join(values(p),','))
        end
    end
    open(joinpath(output_dir,"iterates.csv"),"w") do io
        println(io,"trial,iteration,error,loss,gradient_norm")
        for (trial,r) in enumerate(runs), k in eachindex(r.errors)
            println(io,join((trial,k-1,r.errors[k],r.losses[k],r.gradients[k]),','))
        end
    end
    open(joinpath(output_dir,"runs.csv"),"w") do io
        println(io,"trial,status,iterations,hvp,final_error,final_loss,final_gradient_norm")
        for (trial,r) in enumerate(runs)
            println(io,join((trial,r.status,r.iterations,r.hvp,last(r.errors),last(r.losses),last(r.gradients)),','))
        end
    end
    converged = count(r->r.status==:first_order && last(r.gradients)<=tol,runs)
    summary = """
    FastNCG strongly convex chain experiment
    d=$d, phi=$phi, nu=$nu, trials=$trials, seed=$seed
    x0=$(initial_scale)*randn(MersenneTwister(seed),d); independent draws, no normalization
    gradient tolerance=$tol, max_iter=$max_iter
    solver_kwargs=$solver_kwargs (unspecified options use FastNCGSolver defaults)
    converged=$converged/$trials
    finite positive error pairs=$(length(pairs)), excluded=$excluded
    log base=e, horizontal=log(error_n), vertical=log(error_n+1)
    pooled slope=$(fit.slope), intercept=$(fit.intercept), R2=$(fit.r2)
    iteration range=$(minimum(r.iterations for r in runs)):$(maximum(r.iterations for r in runs))
    maximum final gradient=$(maximum(last(r.gradients) for r in runs))
    Pooled slope alone does not establish per-trajectory local quadratic convergence.
    Inspect local_order in error_pairs.csv; all runs and transient pairs are retained.
    """
    write(joinpath(output_dir,"summary.txt"),summary)
    print(summary)
    if plot
        plot_results(output_dir,fit)
    end
    return (;runs,pairs,fit,converged,excluded,output_dir)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    StronglyConvexExperiment.run_experiment()
end
