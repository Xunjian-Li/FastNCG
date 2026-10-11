module ANCGReference

using LinearAlgebra
using Printf

# Direct translation of ANCG.m and capCG.m at upstream commit
# 45532f314fbf2f0c064f676c64bb9e53db0c6ee6. See ../benchmark/reference/ancg/README.md.
# This reference intentionally retains upstream control flow, including quirks.
const UPSTREAM_COMMIT = "45532f314fbf2f0c064f676c64bb9e53db0c6ee6"

# MATLAB uses process CPU time, not elapsed wall time. BenchmarkTools measures
# the wall time of the adapter independently. C clock units on supported hosts:
_cpu_time() = ccall(:clock, Clong, ()) / (Sys.iswindows() ? 1_000.0 : 1_000_000.0)

function capcg(H, g, e, zta, y_init=nothing)
    U = 0.0
    n = size(H, 1)
    hvp_calls = 0
    function Hv(v)
        hvp_calls += 1
        return H * v
    end
    Hbar(v) = Hv(v) + 2*e*v

    H_bar = H + 2*e*Matrix{Float64}(I, n, n)
    kapa = (U + 2*e)/e
    zta_ht = zta/(3*kapa)
    tau = sqrt(kapa)/(sqrt(kapa)+1)
    T = 4*kapa^4/(1-sqrt(tau))^2

    local j, d, d_type, dHd
    if y_init === nothing
        # MATLAB's scalar zero is broadcast when first added to the vector.
        y0 = zeros(n)
        r0 = g
        p0 = -g
        j = 0
    else
        y0 = y_init
        r0 = g - H_bar*y0
        p0 = -r0
        # Upstream does not initialize j in this branch. Do not silently fix it.
    end

    Hb_p0 = Hbar(p0)
    Hp0 = Hb_p0 - 2*e*p0
    if dot(p0, Hb_p0) < e*norm(p0)^2
        d = p0
        d_type = 2
        # No early return here in the pinned MATLAB source.
    elseif norm(Hp0) > U*norm(p0)
        U = norm(Hp0)/norm(p0)
        kapa = (U+2*e)/e
        zta_ht = zta/(3*kapa)
        tau = sqrt(kapa)/(sqrt(kapa)+1)
        T = 4*kapa^4/(1-sqrt(tau))^2
    end
    y_hty = zeros(n, n+1)
    rj, pj, yj = r0, p0, y0
    y_hty[:, 1] = y0
    Hp_curr = Hv(pj)
    iter = 0
    while true
        iter += 1
        Hp_old = Hp_curr
        Hbp_old = Hp_old + 2*e*pj
        aj = dot(rj, rj)/dot(pj, Hbp_old)
        yj = yj + aj*pj
        rjp1 = rj + aj*Hbp_old
        bj = norm(rjp1)^2/norm(rj)^2
        rj = rjp1
        pj = -rj + bj*pj
        j += 1
        # MATLAB grows a matrix on assignment beyond its current last column.
        if j+1 > size(y_hty, 2)
            y_hty = hcat(y_hty, zeros(n))
        end
        y_hty[:, j+1] = yj
        Hp_curr = Hv(pj)
        if norm(Hp_curr) > U*norm(pj)
            U = norm(Hp_curr)/norm(pj)
            kapa = (U+2*e)/e
            zta_ht = zta/(3*kapa)
            tau = sqrt(kapa)/(sqrt(kapa)+1)
            T = 4*kapa^4/(1-sqrt(tau))^2
        end
        Hyj = Hv(yj)
        if norm(Hyj) > U*norm(yj)
            U = norm(Hyj)/norm(yj)
            kapa = (U+2*e)/e
            zta_ht = zta/(3*kapa)
            tau = sqrt(kapa)/(sqrt(kapa)+1)
            T = 4*kapa^4/(1-sqrt(tau))^2
        end
        Hrj = Hv(rj)
        if norm(Hrj) > U*norm(rj)
            U = norm(Hrj)/norm(rj)
            kapa = (U+2*e)/e
            zta_ht = zta/(3*kapa)
            tau = sqrt(kapa)/(sqrt(kapa)+1)
            T = 4*kapa^4/(1-sqrt(tau))^2
        end
        if dot(yj, Hyj + 2*e*yj) < e*norm(yj)^2
            d, d_type, dHd = yj, 2, dot(yj, Hyj)
            break
        elseif norm(rj) <= zta_ht*norm(r0)
            d, d_type, dHd = yj, 1, dot(yj, Hyj)
            break
        else
            Hbp_curr = Hp_curr + 2*e*pj
            if dot(pj, Hbp_curr) < e*norm(pj)^2
                d, d_type, dHd = pj, 2, dot(pj, Hp_curr)
                break
            elseif norm(rj) > sqrt(T)*tau^(j/2)*norm(r0)
                aj = dot(rj, rj)/dot(pj, Hbp_curr)
                yjp1 = yj + aj*pj
                for i in 0:j-1
                    ydiff = yjp1 - y_hty[:, i+1]
                    Hy = Hv(ydiff)
                    if dot(ydiff, Hy + 2*e*ydiff) < e*norm(ydiff)^2
                        d, d_type, dHd = ydiff, 2, dot(ydiff, Hy)
                        break
                    end
                end
                # As upstream: no fallback if the search assigns no direction.
                break
            end
        end
    end
    return d, d_type, dHd, hvp_calls, iter
end

"""
    optimize(fh, gradh, hessh, x0; kwargs...) -> (x, out)

Reference translation of the pinned authors' MATLAB ANCG code. `hessh(x)`
returns an explicit Hessian matrix. Keyword names match MATLAB options except
`eps` is represented by `eps_tol`. Julia keyword defaults supply the complete
option set (avoiding MATLAB's eager evaluation of missing struct fields).
"""
function optimize(fh, gradh, hessh, x0;
        eps_tol=1e-6, gamma0=1.0, zeta=0.1, eta=0.25, theta=0.5,
        maxit=100_000, maxls=200, verbose=true, gamma_cap=1e12,
        warmstartCG=false, beta=2.0, restart=false)
    tStart = _cpu_time()
    gamma = gamma0
    maxit = round(Int, maxit, RoundNearestTiesAway)
    maxls = round(Int, maxls, RoundNearestTiesAway)
    0 < eta <= 0.5 || error("opts.eta must be in (0, 1/2].")
    0 < zeta < 1 || error("opts.zeta must be in (0,1).")
    0 < theta < 1 || error("opts.theta must be in (0,1).")
    gamma = max(1, gamma)

    x = copy(vec(x0))
    d = 0
    fx = fh(x); fe = 1
    gx = gradh(x); ge = 1
    Hx = hessh(x); He = 1
    fx_hist = [fx]
    gn_hist = [norm(gx)]
    gamma_hist = Float64[gamma]
    alpha_hist = Float64[]
    dtype_hist = Int[]
    stop_reason = ""
    k = 0
    hvp_count = 0
    verbose && println("It  f(x)              ||g||           gamma       type    alpha      j   time  CGiter  hvp")

    while norm(gx) > eps_tol && k < maxit
        k += 1
        local alpha, ftrial, x_new, fx_new, gx_new, Hx_new, inneriter
        ng = norm(gx)
        eps_cg = sqrt(gamma*ng)
        zeta = min(0.5, ng^0.5)
        c_sol = eta*(1-eta)*theta/400
        cmin = c_sol
        if warmstartCG
            # This option has an uninitialized j in upstream capCG. In Julia,
            # expand the initial scalar zero to its vector representation.
            y_init = d isa Number ? fill(Float64(d), length(x)) : d
            d, d_type, dHd, cnt_tmp, _ = capcg(Hx, gx, eps_cg, zeta, y_init)
        else
            d, d_type, dHd, cnt_tmp, inneriter = capcg(Hx, gx, eps_cg, zeta)
        end
        hvp_count += cnt_tmp
        if d_type == 2
            dg = dot(d, gx)
            nd = norm(d)
            if nd == 0 || !isfinite(dHd)
                dk = -gx
                ndk = norm(dk)
            else
                sgn = dg < 0 ? -1 : 1
                dk = -sgn*max(1, 1/gamma)*(abs(dHd)/max(nd^3, floatmin(Float64)))*d
                ndk = norm(dk)
            end
            j = 0; ok = false; f0 = fx
            decBase = (eta/2)*min(1, gamma)*ndk^3
            while j <= maxls
                alpha = theta^j
                ftrial = fh(x + alpha*dk); fe += 1
                if isfinite(ftrial) && ftrial <= f0 - decBase*theta^(2*j)
                    ok = true
                    break
                end
                j += 1
            end
            if !ok
                stop_reason = "Line search failed (NC branch)."
                break
            end
            alpha = theta^j
            x_new = x + alpha*dk
            fx_new = ftrial
            gx_new = gradh(x_new); ge += 1
            Hx_new = hessh(x_new); He += 1
            dtype = 2
            ng_new = norm(gx_new)
            if ng_new > 0.5*ng && alpha < theta/gamma
                gamma = min(beta*gamma, gamma_cap)
            end
        else
            dk = d; ndk = norm(dk)
            f1 = fh(x + dk); fe += 1
            goto_accept = false
            if isfinite(f1) && f1 <= fx
                g1 = gradh(x + dk); ge += 1
                if norm(g1) <= 0.5*ng
                    alpha = 1.0; j = 0
                    x_new = x + dk
                    fx_new = f1
                    gx_new = g1
                    Hx_new = hessh(x_new); He += 1
                    goto_accept = true
                end
            end
            if !goto_accept
                j = 0; ok = false; f0 = fx
                base = eta*sqrt(gamma*ng)*ndk^2
                while j <= maxls
                    alpha = theta^j
                    ftrial = fh(x + alpha*dk); fe += 1
                    if isfinite(ftrial) && ftrial <= f0 - base*theta^j
                        ok = true
                        break
                    end
                    j += 1
                end
                if !ok
                    stop_reason = "Line search failed (SOL branch)."
                    break
                end
                alpha = theta^j
                x_new = x + alpha*dk
                fx_new = ftrial
                gx_new = gradh(x_new); ge += 1
                Hx_new = hessh(x_new); He += 1
            end
            dtype = 1
            ng_new = norm(gx_new)
            if ng_new > 0.5*ng && fx - fx_new < cmin*gamma^(-1/2)*ng^(3/2)
                gamma = min(beta*gamma, gamma_cap)
            end
        end
        x, fx, gx, Hx = x_new, fx_new, gx_new, Hx_new
        push!(fx_hist, fx)
        push!(gn_hist, ng_new)
        push!(gamma_hist, gamma)
        push!(alpha_hist, alpha)
        push!(dtype_hist, dtype)
        if verbose && mod(k, 100) == 0
            @printf("%-3d %-16.8e %-14.3e %-12.3e %-3d %-7.3e %2d %.2f %d %d\n",
                    k, fx, last(gn_hist), gamma, dtype, alpha, j,
                    _cpu_time()-tStart, inneriter, cnt_tmp)
        end
        if mod(k, 20) == 0 && restart
            gamma = gamma0
        end
    end
    if norm(gx) <= eps_tol
        stop_reason = "First-order tolerance satisfied."
    elseif isempty(stop_reason)
        stop_reason = "Max iterations reached."
    end
    out = (iters=k, fevals=fe, gevals=ge, Hevals=He,
           time=_cpu_time()-tStart, fx_hist=fx_hist, gn_hist=gn_hist,
           gamma_hist=gamma_hist, alpha_hist=alpha_hist, dtype_hist=dtype_hist,
           stop=stop_reason, hvp_count=hvp_count)
    return x, out
end

end # module

# NLPModels adapter for the reference algorithm above.

"""
    ancg_optimize(nlp; initial_theta, max_iter=100_000, tol=1e-6, kwargs...)

Benchmark adapter for the explicit-Hessian authors' implementation. Defaults
match ANCG.m; the benchmark overrides only the shared tolerance/iteration limit
and disables printing. No FastNCG objective-difference safeguard is used.

`hess_fn`, if supplied, returns the analytic Hessian. Otherwise the adapter
reconstructs H column-by-column through the existing HVP interface. That cost
is timed and recorded separately in `details.hessian_hvp`. `hvp` includes both
that work and the author's counted CG matrix-vector products. All MATLAB
histories and the exact termination reason are available in `details`.
"""
function ancg_optimize(nlp; initial_theta, max_iter=100_000, tol=1e-6,
        gamma_0=1.0, theta=0.5, eta=0.25, zeta=0.1, max_ls=200,
        gamma_cap=1e12, beta=2.0, warmstartCG=false, restart=false,
        verbose=false, hess_fn=nothing)
    reset!(nlp)
    hessian_hvp = 0
    function hessh(x)
        hess_fn !== nothing && return hess_fn(x)
        n = length(x)
        H = Matrix{Float64}(undef, n, n)
        basis = zeros(n)
        column = zeros(n)
        for j in 1:n
            basis[j] = 1.0
            NLPModels.hprod!(nlp, x, basis, column)
            hessian_hvp += 1
            H[:, j] = column
            basis[j] = 0.0
        end
        return H
    end
    elapsed = @elapsed begin
        x, out = ANCGReference.optimize(
            x -> obj(nlp, x), x -> grad(nlp, x), hessh, initial_theta;
            eps_tol=tol, gamma0=gamma_0, theta=theta, eta=eta, zeta=zeta,
            maxit=max_iter, maxls=max_ls, gamma_cap=gamma_cap, beta=beta,
            warmstartCG=warmstartCG, restart=restart, verbose=verbose)
    end
    details = merge(out, (hessian_hvp=hessian_hvp,
                         upstream_commit=ANCGReference.UPSTREAM_COMMIT))
    return OptResult("ANCG", elapsed, last(out.fx_hist), last(out.gn_hist),
                     out.iters, out.hvp_count+hessian_hvp,
                     out.fevals, out.gevals, x, details)
end
