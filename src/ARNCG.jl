# ============================================================================
# ARNCG — direct Julia translation of authors' MATLAB implementation
# ============================================================================

# Julia translation of the authors' MATLAB ARNCG implementation.
# Source: ../benchmark/reference/arncg/AdapNewtonCG.m and CappedCG.m (see its README).
# Default configuration matches TestCUTEst.m index=2: ARNCG_g, lambda=0, theta=1.
# Public entry point: arncg_official_optimize.

function arncg_official_optimize(nlp; initial_theta,
        max_iter::Int=1_000, tol::Float64=1e-5,
        cg_reltol::Float64=0.01, cg_abstol::Float64=0.01,
        cg_maxiter::Int=length(initial_theta)+2,
        max_omega::Float64=Inf,
        beta::Float64=0.5, mu::Float64=0.3, min_alpha::Float64=0.3,
        gamma::Float64=5.0, tau_minus::Float64=0.3, tau_plus::Float64=1.0,
        theta::Float64=1.0,
        regularization_policy::Symbol=:gradient,
        acceleration_policy::Symbol=:gradient,
        fixed_omega::Float64=sqrt(1e-5),
        fallback_enabled::Bool=false,
        fallback_growth_threshold::Float64=100.0,
        fallback_shrink_threshold::Float64=0.01,
        minimal_norm_d::Union{Nothing,Float64}=2e-16,
        exit_for_many_unchanged_f_and_g::Int=20,
        max_time::Float64=5*3600.0,
        verbose::Bool=false)

    reset!(nlp)
    start = time()
    hvp_c = 0
    grad_evals = 1
    func_evals = 1
    hess_evals = 0
    rejected_step_flag = 0

    # Count actual Hessian-vector products while keeping the official CG logic.
    function hv(x, v)
        hvp_c += 1
        return hprod(nlp, x, v)
    end

    function capped_cg(x, G, reltol, abstol, rho, max_cg)
        Y = zeros(size(G)); Hy = zeros(size(G))
        R = copy(G)
        Hr = hv(x, R)
        P = -R
        Hp = hv(x, P)
        norm_rr0 = dot(R, R)

        U = 0.0
        kappa = (U + 2*rho) / rho
        tau_cg = sqrt(kappa) / (sqrt(kappa) + 1)
        T = 4*kappa^4 / (1 - sqrt(kappa))^2
        dtype = :EMPTY
        norm_rr = norm_rr0
        norm_pbarHp = NaN
        barHp = similar(P)
        it_out = 0

        for it in 0:max_cg
            it_out = it
            if it > 0
                alpha = norm_rr / norm_pbarHp
                Y = Y + alpha * P
                Hy = Hy + alpha * Hp
                R = alpha * barHp + R
                Hr = hv(x, R)
                beta_cg = dot(R,R) / norm_rr
                P = beta_cg * P - R
                Hp = beta_cg * Hp - Hr
            end

            norm_Hp = dot(Hp,Hp); norm_pp = dot(P,P)
            barHp = Hp + 2*rho*P
            norm_pbarHp = dot(P,barHp)
            if sqrt(norm_Hp) > U*sqrt(norm_pp)
                U = sqrt(norm_Hp)/sqrt(norm_pp)
            end
            norm_Hr = dot(Hr,Hr); norm_rr = dot(R,R)
            if sqrt(norm_Hr) > U*sqrt(norm_rr)
                U = sqrt(norm_Hr)/sqrt(norm_rr)
            end
            norm_Hy = dot(Hy,Hy); norm_yy = dot(Y,Y)
            if sqrt(norm_Hy) > U*sqrt(norm_yy)
                U = sqrt(norm_Hy)/sqrt(norm_yy)
            end

            kappa = (U + 2*rho)/rho
            tau_cg = sqrt(kappa)/(sqrt(kappa)+1)
            T = 4*kappa^4/(1-sqrt(kappa))^2
            hat_xi = reltol/(3*kappa)

            norm_ybarHy = dot(Y, Hy + 2*rho*Y)
            if norm_ybarHy < rho*norm_yy
                return copy(Y), sqrt(norm_rr), it_out, :NC
            elseif sqrt(norm_rr) < hat_xi*sqrt(norm_rr0) && sqrt(norm_rr) < abstol
                return copy(Y), sqrt(norm_rr), it_out, :SOL
            elseif norm_pbarHp < rho*norm_pp
                return copy(P), sqrt(norm_rr), it_out, :NC
            elseif sqrt(norm_rr) > sqrt(T)*(tau_cg^(it/2))*sqrt(norm_rr0)
                # Official TestCUTEst uses cg_policy='recompute'.
                alpha = norm_rr/norm_pbarHp
                Y_extra = Y + alpha*P
                HY_extra = Hy + alpha*Hp

                Y = zeros(size(G)); Hy = zeros(size(G))
                R = copy(G); Hr = hv(x,R)
                P = -R; Hp = hv(x,P)
                dtype = :ERR

                for i in 1:(it+1)
                    diff = Y_extra - Y
                    sqrnorm_diff = dot(diff,diff)
                    diff_H_diff = dot(diff, HY_extra-Hy)
                    if diff_H_diff < rho*sqrnorm_diff
                        return copy(diff), sqrt(norm_rr), it_out, :NC
                    end

                    # Preserve the MATLAB source literally: condition is `it > 0`.
                    if it > 0
                        alpha = norm_rr/norm_pbarHp
                        Y = Y + alpha*P
                        Hy = Hy + alpha*Hp
                        R = alpha*barHp + R
                        Hr = hv(x,R)
                        beta_cg = dot(R,R)/norm_rr
                        P = beta_cg*P - R
                        Hp = beta_cg*Hp - Hr
                    end

                    norm_Hp = dot(Hp,Hp); norm_pp = dot(P,P)
                    barHp = Hp + 2*rho*P
                    norm_pbarHp = dot(P,barHp)
                    norm_Hr = dot(Hr,Hr); norm_rr = dot(R,R)
                    norm_Hy = dot(Hy,Hy); norm_yy = dot(Y,Y)
                end
                break
            end
        end

        # Upstream leaves output Z undefined when the historical search fails.
        # Do not invent a direction and then treat ERR as negative curvature.
        dtype == :ERR && throw(UndefVarError(:Z))
        return copy(Y), sqrt(norm_rr), it_out, :SOL
    end

    M = 1.0
    x = copy(initial_theta)
    g = grad(nlp,x)
    f = obj(nlp,x)
    norm_g = norm(g)
    norm_g_min = norm_g
    prev_norm_g = norm_g
    prev_norm_g_min = norm_g_min
    fallback_flag = false

    # Pre-step histories match the authors' records and support parity checks.
    f_hist = Float64[]
    g_hist = Float64[]
    alpha_hist = Float64[]
    M_hist = Float64[]
    cg_hist = Int[]
    dtype_hist = Symbol[]
    it_done = 0

    for it in 1:max_iter
        norm_g < tol && break
        it_done = it
        hess_evals += 1 - rejected_step_flag
        rejected_step_flag = 0

        omega_base = regularization_policy == :gradient ? sqrt(norm_g) :
                     regularization_policy == :minimum_gradient ? sqrt(norm_g_min) :
                     regularization_policy == :fixed ? fixed_omega : error("Unknown regularization policy")

        if !fallback_flag
            omega = acceleration_policy == :gradient ? omega_base*min(1.0,(norm_g/prev_norm_g)^theta) :
                    acceleration_policy == :minimum_gradient ? omega_base*(norm_g_min/prev_norm_g_min)^theta :
                    acceleration_policy == :fixed ? fixed_omega : error("Unknown acceleration policy")
        else
            omega = omega_base
            fallback_flag = false
        end
        omega = min(omega,max_omega)
        cgtol = min(cg_reltol,sqrt(M)*omega)

        tilde_d,cg_norm_rr,cg_it,cg_dtype = capped_cg(x,g,cgtol,cg_abstol,sqrt(M)*omega,cg_maxiter)
        dot_g_tilde_d = dot(g,tilde_d)
        linesearch_failure_flag = false
        smaller_stepsize_flag = false

        if cg_dtype == :SOL
            d = tilde_d; norm_d = norm(d); dot_g_d = dot_g_tilde_d
            alpha = 1.0
            x_new = x; f_new = f
            while alpha > min_alpha
                x_new = x + alpha*d
                f_new = obj(nlp,x_new); func_evals += 1
                f_new <= f + mu*alpha*dot_g_d && break
                alpha *= beta
            end
            hat_alpha = sqrt(omega/sqrt(M)/norm_d)
            if hat_alpha < 1 && alpha <= min_alpha
                alpha = 1.0; smaller_stepsize_flag = true
                while alpha > min_alpha
                    x_new = x + hat_alpha*alpha*d
                    f_new = obj(nlp,x_new); func_evals += 1
                    f_new <= f + mu*hat_alpha*alpha*dot_g_d && break
                    alpha *= beta
                end
                alpha <= min_alpha && (linesearch_failure_flag = true)
                alpha = hat_alpha*alpha
            elseif alpha <= min_alpha
                linesearch_failure_flag = true
            end
        else
            norm_tilde_d = norm(tilde_d)
            normalized_tilde_d = tilde_d/norm_tilde_d
            L_d = abs(dot(hv(x,normalized_tilde_d),normalized_tilde_d))/M
            d = -L_d*sign(dot(g,normalized_tilde_d))*normalized_tilde_d
            norm_d = norm(d); dot_g_d = dot(g,d)
            alpha = 1.0; x_new=x; f_new=f
            while alpha > min_alpha
                x_new = x + alpha*d
                f_new = obj(nlp,x_new); func_evals += 1
                f_new <= f - M*mu*alpha^2*norm_d^3 && break
                alpha *= beta
            end
            alpha <= min_alpha && (linesearch_failure_flag = true)
        end

        M_new = M
        predict_lipcoeff = tau_plus/sqrt(M)
        M_dec_coeff = mu*tau_minus
        gradient_computed = false
        g_new = g

        if cg_dtype == :SOL
            if alpha < 1
                predict_descent = predict_lipcoeff*omega^3
                f < beta*mu*predict_descent + f_new && (M_new=M*gamma)
            else
                g_new = grad(nlp,x_new); grad_evals += 1; gradient_computed=true
                predict_descent = predict_lipcoeff*min(omega^3,norm(g_new)^2/omega)
                M_dec_coeff *= 4/33
                f < (4/33)*mu*predict_descent + f_new && (M_new=M*gamma)
            end
        else
            predict_descent = predict_lipcoeff*omega^3
            f < ((1-2*mu)*beta)^2*mu*predict_descent + f_new && (M_new=M*gamma)
        end

        if f > M_dec_coeff*predict_lipcoeff*omega_base^3 + f_new
            M_new = M/gamma
        end
        Delta = f-f_new

        if linesearch_failure_flag
            M_new=M*gamma; x_new=x; f_new=f; g_new=g
            rejected_step_flag=1; gradient_computed=true
        end

        if fallback_enabled && theta > 0
            if norm_g < prev_norm_g*fallback_shrink_threshold
                if !gradient_computed
                    g_new=grad(nlp,x_new); grad_evals+=1; gradient_computed=true
                end
                next_norm_g=norm(g_new)
                if next_norm_g > norm_g*fallback_growth_threshold
                    fallback_flag=true; rejected_step_flag=1
                    x_new=x; g_new=g; f_new=f; M_new=M
                end
            end
        end

        push!(f_hist,f); push!(g_hist,norm_g)
        push!(alpha_hist,alpha); push!(M_hist,M)
        push!(cg_hist,cg_it); push!(dtype_hist,cg_dtype)
        if exit_for_many_unchanged_f_and_g > 0 && length(f_hist) > exit_for_many_unchanged_f_and_g
            j=length(f_hist)-exit_for_many_unchanged_f_and_g
            if f_hist[end] == f_hist[j] && g_hist[end] == g_hist[j]
                break
            end
        end

        x=x_new; M=M_new
        if !gradient_computed
            grad_evals += 1; g=grad(nlp,x)
        else
            g=g_new
        end
        prev_norm_g=norm_g; prev_norm_g_min=norm_g_min
        norm_g=norm(g); norm_g_min=min(norm_g,norm_g_min)
        f=f_new

        isnan(f) && break
        M > 1e40 && break
        time()-start > max_time && break
        minimal_norm_d !== nothing && norm_d < minimal_norm_d && break
    end

    details = (Hevals=hess_evals, fx_hist=f_hist, gn_hist=g_hist,
               alpha_hist=alpha_hist, M_hist=M_hist, cg_hist=cg_hist,
               dtype_hist=dtype_hist)
    return OptResult("ARNCG",time()-start,f,norm_g,it_done,hvp_c,func_evals,grad_evals,copy(x),details)
end
