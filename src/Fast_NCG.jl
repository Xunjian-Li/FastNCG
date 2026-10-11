using LinearAlgebra
using NLPModels, Printf

import NLPModels: obj, grad!, hprod!

export FastNCGSolver, ClosureNLPModel, optimize!

"""
    ClosureNLPModel(x0, obj_fn, grad_fn, hprod_fn; objdiff_fn=nothing, name="ClosureNLPModel")

Minimal `NLPModels.jl` wrapper for an objective, in-place gradient,
in-place Hessian-vector product, and optional stable objective-difference callback.

Callbacks:
    obj_fn(x)
    grad_fn(g, x)
    hprod_fn(Hv, x, v)
    objdiff_fn(x, trial) -> (f(trial) - f(x), auxiliary)
"""
mutable struct ClosureNLPModel{OF,GF,HF,DF} <: AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModelMeta{Float64,Vector{Float64}}
    counters::NLPModels.Counters
    obj_fn::OF
    grad_fn::GF
    hprod_fn::HF
    objdiff_fn::DF
end

function ClosureNLPModel(x0::Vector{Float64}, obj_fn, grad_fn, hprod_fn; objdiff_fn=nothing, name::String="ClosureNLPModel")
    meta = NLPModelMeta(length(x0); x0=copy(x0), name=name)
    counters = NLPModels.Counters()
    return ClosureNLPModel(meta, counters, obj_fn, grad_fn, hprod_fn, objdiff_fn)
end

function NLPModels.obj(nlp::ClosureNLPModel, x)
    nlp.counters.neval_obj += 1
    return nlp.obj_fn(x)
end

function NLPModels.grad!(nlp::ClosureNLPModel, x::AbstractVector, g::AbstractVector)
    nlp.counters.neval_grad += 1
    nlp.grad_fn(g, x)
    return g
end

function NLPModels.hprod!(nlp::ClosureNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; kwargs...)
    nlp.counters.neval_hprod += 1
    nlp.hprod_fn(Hv, x, v)
    return Hv
end

function _objective_difference(nlp::ClosureNLPModel, x, trial)
    nlp.objdiff_fn === nothing && return nothing
    result = nlp.objdiff_fn(x, trial)
    return result isa Tuple ? result[1] : result
end

"""
    FastNCGSolver(; kwargs...)

Sparse regularized Newton-CG solver using a Lanczos/Krylov reduced model
and a scalar target-ratio equation. `max_regularization_retries` limits sigma
increases per outer iteration independently of `max_krylov` (zero allows only
the initial sigma). After `optimize!`, `status` is `:first_order`, `:max_iter`,
`:inner_failure`, or `:numerical_stagnation`. Without a finite `objdiff_fn` result, trial acceptance
requires a strictly negative computed objective difference.
"""
mutable struct FastNCGSolver
    sigma_0_ref::Float64
    sigma_min::Float64
    rho::Float64
    beta::Float64
    gamma::Float64
    tau::Float64
    max_outer::Int
    max_krylov::Int
    max_newton::Int
    max_regularization_retries::Int
    tol_newton::Float64
    verbose::Bool

    hvp_count::Int
    grad_count::Int
    func_count::Int
    krylov_extensions::Int
    ratio_increases::Int
    newton_iters_total::Int
    accepted_kappa::Vector{Float64}
    attained_dims::Vector{Int}

    loss_history::Vector{Float64}
    grad_norm_history::Vector{Float64}
    time_history::Vector{Float64}
    total_time::Float64
    status::Symbol
end

function FastNCGSolver(; sigma_0_ref::Float64=1.0, sigma_min::Float64=1e-8, rho::Float64=2.0,
                beta::Float64=0.25, gamma::Float64=1e-3, tau::Float64=0.1,
                max_outer::Int=500, max_krylov::Int=0, max_newton::Int=10,
                tol_newton::Float64=1e-3, verbose::Bool=false,
                max_regularization_retries::Int=50)
    @assert max_regularization_retries >= 0
    @assert 0 < beta < 0.5
    @assert 0 < gamma < tau < 1
    @assert rho > 1
    @assert sigma_0_ref >= sigma_min > 0

    return FastNCGSolver(
        sigma_0_ref, sigma_min, rho, beta, gamma, tau,
        max_outer, max_krylov, max_newton, max_regularization_retries, tol_newton, verbose,
        0, 0, 0, 0, 0, 0, Float64[], Int[],
        Float64[], Float64[], Float64[], 0.0, :not_started
    )
end

function _reset_stats!(s::FastNCGSolver)
    s.hvp_count = 0
    s.grad_count = 0
    s.func_count = 0
    s.krylov_extensions = 0
    s.ratio_increases = 0
    s.newton_iters_total = 0
    empty!(s.accepted_kappa)
    empty!(s.attained_dims)
    empty!(s.loss_history)
    empty!(s.grad_norm_history)
    empty!(s.time_history)
    s.total_time = 0.0
    s.status = :running
end

function _lanczos_extension_qdirect!(s::FastNCGSolver, nlp::N, theta,
                                     q_prev::AbstractVector{Float64}, q_curr::AbstractVector{Float64},
                                     delta_prev::Float64, Hq_work::Vector{Float64},
                                     w_work::Vector{Float64}, q_next::AbstractVector{Float64}) where {N<:ClosureNLPModel}
    hprod!(nlp, theta, q_curr, Hq_work)
    s.hvp_count += 1

    @inbounds @simd for i in eachindex(w_work, Hq_work, q_prev)
        w_work[i] = Hq_work[i] - delta_prev * q_prev[i]
    end

    theta_j = dot(q_curr, w_work)

    @inbounds @simd for i in eachindex(w_work, q_curr)
        w_work[i] -= theta_j * q_curr[i]
    end

    coeff = dot(w_work, q_curr)
    theta_j += coeff

    @inbounds @simd for i in eachindex(w_work, q_curr)
        w_work[i] -= coeff * q_curr[i]
    end

    delta_j = norm(w_work)
    delta_j < 1e-14 && return theta_j, 0.0, true

    @inbounds @simd for i in eachindex(q_next, w_work)
        q_next[i] = w_work[i] / delta_j
    end

    return theta_j, delta_j, false
end

function _solve_tridiagonal_ws!(y::AbstractVector{Float64}, diag::AbstractVector{Float64},
                                offdiag::AbstractVector{Float64}, rhs::AbstractVector{Float64},
                                c_prime::AbstractVector{Float64}, d_prime::AbstractVector{Float64},
                                pivots::Union{Nothing,AbstractVector{Float64}}=nothing)
    k = length(rhs)
    k == 0 && return y

    denom = abs(diag[1]) > 1e-15 ? diag[1] : 1e-15
    pivots === nothing || (pivots[1] = denom)
    c_prime[1] = k > 1 ? offdiag[1] / denom : 0.0
    d_prime[1] = rhs[1] / denom

    @inbounds for i in 2:k
        denom = diag[i] - offdiag[i-1] * c_prime[i-1]
        abs(denom) < 1e-15 && (denom = 1e-15)
        pivots === nothing || (pivots[i] = denom)
        c_prime[i] = i < k ? offdiag[i] / denom : 0.0
        d_prime[i] = (rhs[i] - offdiag[i-1] * d_prime[i-1]) / denom
    end

    y[k] = d_prime[k]
    @inbounds for i in (k-1):-1:1
        y[i] = d_prime[i] - c_prime[i] * y[i+1]
    end

    return y
end

# Reuse the pivots and upper factors from the preceding solve at the same lambda.
function _solve_factored_tridiagonal_ws!(y, pivots, offdiag, rhs, c_prime, d_prime)
    k = length(rhs)
    k == 0 && return y
    d_prime[1] = rhs[1] / pivots[1]
    @inbounds for i in 2:k
        d_prime[i] = (rhs[i] - offdiag[i-1] * d_prime[i-1]) / pivots[i]
    end
    y[k] = d_prime[k]
    @inbounds for i in (k-1):-1:1
        y[i] = d_prime[i] - c_prime[i] * y[i+1]
    end
    return y
end

function _min_eig_tridiag(diag::Vector{Float64}, offdiag::Vector{Float64})
    k = length(diag)
    k == 0 && return 0.0
    k == 1 && return diag[1]

    T = SymTridiagonal(@view(diag[1:k]), @view(offdiag[1:k-1]))
    try
        return minimum(eigvals(T))
    catch
        return 0.0
    end
end

function _eval_F_only_ws!(diag_shifted::AbstractVector{Float64}, u::AbstractVector{Float64},
                          c_prime::AbstractVector{Float64}, d_prime::AbstractVector{Float64},
                          diag::AbstractVector{Float64}, offdiag::AbstractVector{Float64},
                          e1::AbstractVector{Float64}, lam::Float64,
                          sigma::Float64, g_norm::Float64)
    j = length(diag)

    @inbounds for i in 1:j
        di = diag[i] + lam
        diag_shifted[i] = di
        di <= 1e-15 && return nothing, nothing
    end

    try
        # Store elimination pivots in the disposable shifted-diagonal workspace.
        _solve_tridiagonal_ws!(u, diag_shifted, offdiag, e1, c_prime, d_prime, diag_shifted)
    catch
        return nothing, nothing
    end

    u_norm = norm(u)
    !isfinite(u_norm) && return nothing, nothing

    return lam - sigma * g_norm * u_norm, u_norm
end

function _solve_target_ratio_ws!(s::FastNCGSolver, diag::Vector{Float64}, offdiag::Vector{Float64},
                                 sigma::Float64, g_norm::Float64, a_j::Float64,
                                 diag_shifted_buf::Vector{Float64}, e1_buf::Vector{Float64},
                                 u_buf::Vector{Float64}, w_buf::Vector{Float64},
                                 c_prime_buf::Vector{Float64}, d_prime_buf::Vector{Float64},
                                 y_result_buf::Vector{Float64})
    j = length(diag)
    j == 0 && return nothing, nothing, false

    off = @view offdiag[1:max(j-1, 0)]
    diag_shifted = @view diag_shifted_buf[1:j]
    e1 = @view e1_buf[1:j]
    u = @view u_buf[1:j]
    w = @view w_buf[1:j]
    c_prime = @view c_prime_buf[1:j]
    d_prime = @view d_prime_buf[1:j]
    y_result = @view y_result_buf[1:j]

    fill!(e1, 0.0)
    e1[1] = 1.0

    asymptotic_scale = sqrt(max(sigma, 0.0)) * sqrt(max(g_norm, 0.0))
    base_gap = max(1e-6, 0.01 * max(1.0, abs(a_j)))
    gap = max(base_gap, asymptotic_scale)
    lam = a_j + gap
    found_left_start = false

    for _ in 1:32
        F_val, _ = _eval_F_only_ws!(diag_shifted, u, c_prime, d_prime,
                                     diag, off, e1, lam, sigma, g_norm)

        if F_val !== nothing && isfinite(F_val) && F_val < 0
            found_left_start = true
            break
        end

        gap *= 0.5
        lam_new = a_j + gap
        !(lam_new > a_j) && break
        lam = lam_new
    end

    found_left_start || return nothing, nothing, false

    for it in 1:s.max_newton
        F_val, u_norm = _eval_F_only_ws!(diag_shifted, u, c_prime, d_prime,
                                          diag, off, e1, lam, sigma, g_norm)
        F_val === nothing && return nothing, nothing, false

        try
            # _eval_F_only_ws! already computed u and left its factors in
            # diag_shifted/c_prime. Only the second right-hand side remains.
            _solve_factored_tridiagonal_ws!(w, diag_shifted, off, u, c_prime, d_prime)
        catch
            return nothing, nothing, false
        end

        F_prime = 1.0 + sigma * g_norm * dot(u, w) / max(u_norm, 1e-30)
        F_prime < 1e-15 && return nothing, nothing, false

        step = F_val / F_prime
        lam_new = lam - step
        lam_new <= a_j + 1e-15 && (lam_new = a_j + 0.5 * (lam - a_j))

        if abs(step) < s.tol_newton * max(1.0, abs(lam))
            s.newton_iters_total += it

            @inbounds for i in 1:j
                diag_shifted[i] = diag[i] + lam_new
            end

            try
                _solve_tridiagonal_ws!(u, diag_shifted, off, e1, c_prime, d_prime)
            catch
                return nothing, nothing, false
            end

            @inbounds for i in 1:j
                y_result[i] = g_norm * u[i]
            end

            # Borrowed workspace: the caller consumes y before the next solve.
            return lam_new, y_result, true
        end

        lam = lam_new
    end

    s.newton_iters_total += s.max_newton

    @inbounds for i in 1:j
        diag_shifted[i] = diag[i] + lam
    end

    try
        _solve_tridiagonal_ws!(u, diag_shifted, off, e1, c_prime, d_prime)
    catch
        return nothing, nothing, false
    end

    @inbounds for i in 1:j
        y_result[i] = g_norm * u[i]
    end

    return lam, y_result, true
end

function _projection_ratio_from_mat(Q_mat::AbstractMatrix{Float64}, g_new::Vector{Float64},
                                    proj::AbstractVector{Float64})
    g_norm_sq = dot(g_new, g_new)
    g_norm_sq < 1e-30 && return 1.0

    mul!(proj, Q_mat', g_new)
    proj_sq = dot(proj, proj)
    proj_sq > g_norm_sq && (proj_sq = g_norm_sq)

    return min(sqrt(max(0.0, proj_sq) / g_norm_sq), 1.0)
end

# Views are rebuilt by the caller after extension; copy only initialized columns.
function _ensure_krylov_capacity(Q::Matrix{Float64}, needed::Int, limit::Int)
    needed <= size(Q, 2) && return Q
    columns = min(limit, max(needed, 2size(Q, 2)))
    grown = Matrix{Float64}(undef, size(Q, 1), columns)
    copyto!(@view(grown[:, 1:needed-1]), @view(Q[:, 1:needed-1]))
    return grown
end

function _extend_krylov_qdirect!(s::FastNCGSolver, nlp::N, x_n, Q::Matrix{Float64}, j::Int,
                                 diag_list::Vector{Float64}, offdiag_list::Vector{Float64},
                                 delta_j::Float64, Hq_work::Vector{Float64},
                                 w_work::Vector{Float64}) where {N<:ClosureNLPModel}
    delta_j <= 0 && return 0.0, j, true

    j_new = j + 1
    q_prev = @view Q[:, j_new-1]
    q_curr = @view Q[:, j_new]
    q_next = @view Q[:, j_new+1]

    theta_new, delta_new, broke_new = _lanczos_extension_qdirect!(
        s, nlp, x_n, q_prev, q_curr, delta_j, Hq_work, w_work, q_next
    )

    push!(diag_list, theta_new)

    if broke_new
        return 0.0, j_new, true
    else
        push!(offdiag_list, delta_new)
        return delta_new, j_new, false
    end
end

"""
    optimize!(solver, nlp; initial_theta=nothing, max_iter=nothing, tol=1e-5)

Run FastNCG on an `NLPModels.jl` model. The default initial point is a copy of
`nlp.meta.x0`. An unchanged trial point stops with `:numerical_stagnation`.
"""
function optimize!(s::FastNCGSolver, nlp; initial_theta::Union{Nothing,Vector{Float64}}=nothing,
                   max_iter::Union{Nothing,Int}=nothing, tol::Float64=1e-5)
    _reset_stats!(s)

    dim = nlp.meta.nvar
    max_outer = max_iter === nothing ? s.max_outer : max_iter
    max_krylov = s.max_krylov == 0 ? dim : min(s.max_krylov, dim)
    eps_stop = tol

    x_n = initial_theta === nothing ? copy(nlp.meta.x0) : copy(initial_theta)

    Q = Matrix{Float64}(undef, dim, min(max_krylov, 16) + 1)
    lanczos_Hq = Vector{Float64}(undef, dim)
    lanczos_w = Vector{Float64}(undef, dim)

    tr_diag_shifted = Vector{Float64}(undef, max_krylov)
    tr_e1 = zeros(Float64, max_krylov)
    tr_u = Vector{Float64}(undef, max_krylov)
    tr_w = Vector{Float64}(undef, max_krylov)
    tr_cprime = Vector{Float64}(undef, max_krylov)
    tr_dprime = Vector{Float64}(undef, max_krylov)
    tr_yresult = Vector{Float64}(undef, max_krylov)

    x_trial_work = Vector{Float64}(undef, dim)
    g_trial_work = Vector{Float64}(undef, dim)
    q_prev_work = zeros(dim)
    q_curr_work = Vector{Float64}(undef, dim)
    reduced_matvec_work = Vector{Float64}(undef, dim)
    projection_work = Vector{Float64}(undef, max_krylov)
    last_v = zeros(dim)
    last_g_trial = zeros(dim)
    diag_list = Float64[]
    offdiag_list = Float64[]
    sizehint!(diag_list, min(max_krylov, 16))
    sizehint!(offdiag_list, min(max_krylov, 16))

    start_time = time()

    f_n = obj(nlp, x_n)
    s.func_count += 1

    g_n = Vector{Float64}(undef, dim)
    grad!(nlp, x_n, g_n)
    s.grad_count += 1

    g_norm = norm(g_n)
    g_norm_sq = g_norm^2

    push!(s.loss_history, f_n)
    push!(s.grad_norm_history, g_norm)
    push!(s.time_history, time() - start_time)

    sigma_ref = s.sigma_0_ref

    for n in 1:max_outer
        g_norm_sq <= eps_stop^2 && break

        q_prev = q_prev_work
        q_curr = q_curr_work
        fill!(q_prev, 0.0)

        denom = max(g_norm, 1e-15)
        @inbounds for i in eachindex(q_curr, g_n)
            q_curr[i] = -g_n[i] / denom
        end
        copyto!(@view(Q[:, 1]), q_curr)

        empty!(diag_list)
        empty!(offdiag_list)

        theta_1, delta_1, broke_1 = _lanczos_extension_qdirect!(
            s, nlp, x_n, q_prev, q_curr, 0.0,
            lanczos_Hq, lanczos_w, @view(Q[:, 2])
        )

        push!(diag_list, theta_1)

        local delta_j::Float64
        if broke_1
            delta_j = 0.0
        else
            push!(offdiag_list, delta_1)
            delta_j = delta_1
        end

        j = 1
        Q_mat = @view Q[:, 1:j]
        a_j = max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
        sigma = sigma_ref
        accepted = false

        last_kappa = 0.0
        last_f_trial = 0.0
        last_g_trial_norm = 0.0

        regularization_retries = 0
        inner_iter = 0

        while true
            inner_iter += 1
            isempty(diag_list) && break

            lam, y_j, ok = _solve_target_ratio_ws!(
                s, diag_list, offdiag_list, sigma, g_norm, a_j,
                tr_diag_shifted, tr_e1, tr_u, tr_w,
                tr_cprime, tr_dprime, tr_yresult
            )

            if !ok || lam === nothing || y_j === nothing
                if delta_j > 0 && j < max_krylov
                    Q = _ensure_krylov_capacity(Q, j + 2, max_krylov + 1)
                    delta_j, j, _ = _extend_krylov_qdirect!(
                        s, nlp, x_n, Q, j, diag_list, offdiag_list,
                        delta_j, lanczos_Hq, lanczos_w
                    )
                    s.krylov_extensions += 1
                    Q_mat = @view Q[:, 1:j]
                    a_j = max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
                else
                    regularization_retries >= s.max_regularization_retries && break
                    regularization_retries += 1
                    sigma *= s.rho
                    s.ratio_increases += 1
                end
                continue
            end

            v = reduced_matvec_work
            mul!(v, Q_mat, y_j)
            v_norm = norm(v)

            @inbounds @simd for i in eachindex(x_n, v)
                x_trial_work[i] = x_n[i] + v[i]
            end

            if x_trial_work == x_n
                s.status = :numerical_stagnation
                break
            end

            if v_norm < 1e-15
                if delta_j > 0 && j < max_krylov
                    Q = _ensure_krylov_capacity(Q, j + 2, max_krylov + 1)
                    delta_j, j, _ = _extend_krylov_qdirect!(
                        s, nlp, x_n, Q, j, diag_list, offdiag_list,
                        delta_j, lanczos_Hq, lanczos_w
                    )
                    s.krylov_extensions += 1
                    Q_mat = @view Q[:, 1:j]
                    a_j = max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
                else
                    regularization_retries >= s.max_regularization_retries && break
                    regularization_retries += 1
                    sigma *= s.rho
                    s.ratio_increases += 1
                end
                continue
            end

            kappa_cur = lam / max(v_norm, 1e-15)

            f_trial = obj(nlp, x_trial_work)
            s.func_count += 1

            grad!(nlp, x_trial_work, g_trial_work)
            s.grad_count += 1
            g_trial_norm = norm(g_trial_work)

            if !isfinite(f_trial) || !isfinite(g_trial_norm)
                regularization_retries >= s.max_regularization_retries && break
                regularization_retries += 1
                sigma *= s.rho
                s.ratio_increases += 1
                continue
            end

            rho_proj = _projection_ratio_from_mat(Q_mat, g_trial_work, @view(projection_work[1:j]))
            proj_ok = rho_proj >= s.tau || g_trial_norm < 1e-30
            grad_ok = lam * v_norm >= s.gamma * g_trial_norm
            descent_ok = false
            use_objdiff = false

            # Only evaluate the difference when it can affect acceptance.
            if proj_ok && grad_ok
                required = s.beta * lam * v_norm^2

                # Prefer the stable problem-specific objective difference.
                delta_f = _objective_difference(nlp, x_n, x_trial_work)

                local descent_margin::Float64
                if delta_f !== nothing && isfinite(delta_f)
                    use_objdiff = true
                    descent_margin = delta_f + required
                    descent_ok = isfinite(descent_margin) && descent_margin <= 0.0
                else
                    use_objdiff = false
                    # A decreasing quadratic model does not certify actual descent.
                    # If subtraction cannot resolve a decrease, reject the trial;
                    # supply objdiff_fn for accurate decisions near convergence.
                    actual_delta = f_trial - f_n
                    descent_margin = actual_delta + required
                    descent_ok = actual_delta < 0.0 &&
                                 isfinite(descent_margin) && descent_margin <= 0.0
                end

                if s.verbose && use_objdiff
                    Printf.@printf(
                        "objdiff: outer=%d inner=%d j=%d |v|=%.3e req=%.3e delta=%.3e margin=%.3e\n",
                        n, inner_iter, j, v_norm, required, delta_f, descent_margin
                    )
                end
            end

            if proj_ok && descent_ok && grad_ok
                if s.verbose
                    println(
                        "ACCEPT outer=", n,
                        " j=", j,
                        " |g|=", g_norm,
                        " sigma_ref=", sigma_ref,
                        " sigma=", sigma,
                        " lambda=", lam,
                        " rho_proj=", rho_proj,
                        " kappa/sigma=", kappa_cur / sigma,
                        " objdiff=", use_objdiff
                    )
                end

                accepted = true
                last_kappa = kappa_cur
                copyto!(last_v, v)
                last_f_trial = f_trial
                copyto!(last_g_trial, g_trial_work)
                last_g_trial_norm = g_trial_norm
                break
            end

            if !proj_ok && delta_j > 0 && j < max_krylov
                projection_fraction = rho_proj / max(s.tau, eps(Float64))
                projection_batch = projection_fraction >= 0.75 ? 1 :
                                   projection_fraction >= 0.25 ? 3 : 5

                for _ in 1:projection_batch
                    (j >= max_krylov || delta_j <= 0.0) && break

                    Q = _ensure_krylov_capacity(Q, j + 2, max_krylov + 1)
                    delta_j, j, broke = _extend_krylov_qdirect!(
                        s, nlp, x_n, Q, j, diag_list, offdiag_list,
                        delta_j, lanczos_Hq, lanczos_w
                    )
                    s.krylov_extensions += 1
                    broke && break
                end

                Q_mat = @view Q[:, 1:j]
                a_j = max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
                continue
            end

            regularization_retries >= s.max_regularization_retries && break
            regularization_retries += 1
            sigma *= s.rho
            s.ratio_increases += 1
        end

        if accepted
            kappa_n = last_kappa

            push!(s.accepted_kappa, kappa_n)
            push!(s.attained_dims, j)

            @inbounds @simd for i in eachindex(x_n, last_v)
                x_n[i] += last_v[i]
            end

            f_n = last_f_trial
            copyto!(g_n, last_g_trial)
            g_norm = last_g_trial_norm
            g_norm_sq = g_norm^2
            sigma_ref = max(s.sigma_min, kappa_n / s.rho)
        else
            s.status == :running && (s.status = :inner_failure)
            s.verbose && println(
                "STOP status=", s.status, " outer=", n,
                " |g|=", g_norm,
                " sigma_ref=", sigma_ref,
                " sigma=", sigma,
                " j=", j
            )
            break
        end

        push!(s.loss_history, f_n)
        push!(s.grad_norm_history, g_norm)
        push!(s.time_history, time() - start_time)
    end

    if s.status == :running
        s.status = g_norm_sq <= eps_stop^2 ? :first_order : :max_iter
    end
    s.total_time = time() - start_time
    return x_n
end
