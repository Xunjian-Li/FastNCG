using LinearAlgebra, Random, Printf
using FastNCG: ANCGReference
BLAS.set_num_threads(1)

function write_csv(path, X)
    X = X isa AbstractVector ? reshape(X, :, 1) : X
    open(path, "w") do io
        for i in axes(X, 1)
            for j in axes(X, 2)
                j > first(axes(X, 2)) && print(io, ',')
                @printf(io, "%.17g", X[i, j])
            end
            println(io)
        end
    end
end

function callbacks(name, A, b)
    if name == "quadratic"
        return x -> sum(abs2, x)/2, x -> copy(x), x -> Matrix{Float64}(I, length(x), length(x))
    elseif startswith(name, "fail_")
        h0 = name == "fail_nc" ? -4.0 : 0.0
        return x -> x[1]+h0*x[1]^2/2+1000x[1]^4,
               x -> [1+h0*x[1]+4000x[1]^3],
               x -> reshape([h0+12000x[1]^2], 1, 1)
    elseif name == "rosenbrock"
        function rosen_f(x)
            value = 0.0
            for i in 1:length(x)-1
                r = x[i+1]-x[i]^2
                value += 100r^2+(1-x[i])^2
            end
            return value
        end
        function rosen_g(x)
            result = zeros(length(x))
            for i in 1:length(x)-1
                r = x[i+1]-x[i]^2
                result[i] += -400x[i]*r+2(x[i]-1)
                result[i+1] += 200r
            end
            return result
        end
        function rosen_H(x)
            result = zeros(length(x), length(x))
            for i in 1:length(x)-1
                result[i,i] += 1200x[i]^2-400x[i+1]+2
                result[i,i+1] += -400x[i]
                result[i+1,i] += -400x[i]
                result[i+1,i+1] += 200
            end
            return result
        end
        return rosen_f,rosen_g,rosen_H
    elseif name == "polytope"
        poly_r(x) = max.(A*x-b, 0.0)
        return x -> sum(poly_r(x).^4), x -> A'*(4poly_r(x).^3), x -> A'*((12poly_r(x).^2).*A)
    elseif name == "tregression"
        treg_r(x) = A*x-b
        nu = 0.1
        return x -> sum(log1p.(treg_r(x).^2/nu)),
               x -> A'*(2treg_r(x)./(nu .+ treg_r(x).^2)),
               x -> A'*((2(nu .- treg_r(x).^2)./(nu .+ treg_r(x).^2).^2).*A)
    elseif name == "logsumexp"
        rho = 0.05
        function w(x)
            z = A*x-b
            u = exp.((z .- maximum(z))/rho)
            return u/sum(u)
        end
        function lse_f(x)
            z = A*x-b; zmax = maximum(z)
            return zmax+rho*log(sum(exp.((z .- zmax)/rho)))
        end
        function lse_H(x)
            weights = w(x); a = A'*weights
            return (A'*(weights.*A)-a*a')/rho
        end
        return lse_f, x -> A'*w(x), lse_H
    end
    error(name)
end

function main(root)
    mkpath(root)
    for name in ("quadratic", "fail_sol", "fail_nc", "rosenbrock", "polytope", "tregression", "logsumexp")
        folder = joinpath(root, name); mkpath(folder)
        rng = MersenneTwister(42)
        A = zeros(1,1); b = zeros(1)
        if name == "polytope"
            n,m=400,500
            A=randn(rng,m,n)/sqrt(n); b=randn(rng,m); x0=zeros(n)
        elseif name == "tregression"
            n,m=100,200
            truth=randn(rng,n); A=randn(rng,m,n)
            for j in 2:n
                A[:,j] .+= 0.3 .* A[:,j-1]
            end
            A ./= sqrt(1+0.3^2)
            noise=0.5randn(rng,m); chi=2rand(rng,m)
            b=A*truth+noise./sqrt.(chi); x0=A\b
        elseif name == "logsumexp"
            n,m=200,1000
            A=randn(rng,m,n)/sqrt(n); b=randn(rng,m); x0=zeros(n)
        else
            x0 = name == "quadratic" ? [1.0] : zeros(name == "rosenbrock" ? 100 : 1)
        end
        write_csv(joinpath(folder,"A.csv"),A)
        write_csv(joinpath(folder,"b.csv"),b)
        write_csv(joinpath(folder,"x0.csv"),x0)
        f,g,H=callbacks(name,A,b)
        x,out=ANCGReference.optimize(f,g,H,x0; eps_tol=1e-7,maxit=10000,
                                     maxls=startswith(name,"fail_") ? 0 : 200,verbose=false)
        write_csv(joinpath(folder,"julia_x.csv"),x)
        write_csv(joinpath(folder,"julia_history.csv"),hcat(out.fx_hist,out.gn_hist,out.gamma_hist))
        write_csv(joinpath(folder,"julia_steps.csv"),hcat(out.alpha_hist,out.dtype_hist))
        write_csv(joinpath(folder,"julia_counts.csv"),[out.iters,out.fevals,out.gevals,out.Hevals,out.hvp_count])
        write(joinpath(folder,"julia_stop.txt"),out.stop)
        println(name," Julia: ",out.iters," iterations, ",out.stop)
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
