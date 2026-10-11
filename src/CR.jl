# ============================================================================
# CR — Classical Cubic Regularization (Nesterov & Polyak, 2006)
# ============================================================================
# The cubic model is
#
#   m_k(s) = f_k + g_k' s + 1/2 s' H_k s + (M_k / 6) ||s||^3.
#
# At each outer iteration we compute a GLOBAL minimizer of the cubic model.
# Following Section 5.1 of Nesterov & Polyak (2006), the subproblem is reduced
# to a one-dimensional secular equation after an eigendecomposition of H_k.
# The hard case (the solution lies at the left boundary of the secular domain)
# is handled explicitly.
#
# For the practical line search we use strategy (5.7) of Nesterov & Polyak:
# double M_k until the trial decreases f, accept x_{k+1}=T_{M_k}(x_k), and
# then set M_{k+1}=max(M_k/2, L0).  This is not an ARC ratio test.

function _cr_full_hessian(nlp, x)
    n = length(x)
    H = Matrix{Float64}(undef, n, n)
    e = zeros(Float64, n)

    for j in 1:n
        e[j] = 1.0
        H[:, j] = hprod(nlp, x, e)
        e[j] = 0.0
    end

    # Remove tiny asymmetry caused by floating-point HVP arithmetic.
    H .= 0.5 .* (H .+ H')
    return H, n
end

function _cr_cubic_global_step_from_eigen(
    eig::Eigen{Float64,Float64,Matrix{Float64},Vector{Float64}},
    g::Vector{Float64},
    M::Float64;
    secular_tol::Float64 = 1e-12,
    max_secular::Int = 100,
)
    M > 0.0 || error("CR requires M > 0.")

    d = eig.values
    U = eig.vectors
    gt = U' * g
    n = length(d)

    dmin = d[1]
    lambda_lb = max(0.0, -dmin)

    # Scale-aware tolerance for detecting the eigenspace associated with dmin.
    eig_tol = 100.0 * eps(Float64) * max(1.0, maximum(abs, d))
    g_tol = 100.0 * eps(Float64) * max(1.0, norm(g))

    # At the left boundary lambda_lb, H + lambda I is PSD and may be singular.
    # First test whether the boundary solution is the hard case.
    denom_lb = d .+ lambda_lb
    y_pinv = zeros(Float64, n)
    singular_with_gradient = false

    @inbounds for i in 1:n
        if abs(denom_lb[i]) <= eig_tol
            if abs(gt[i]) > g_tol
                singular_with_gradient = true
            end
        else
            y_pinv[i] = -gt[i] / denom_lb[i]
        end
    end

    r_lb = 2.0 * lambda_lb / M
    y_pinv_norm = norm(y_pinv)

    # Hard case: g is orthogonal to the leftmost eigenspace and the
    # pseudoinverse component is not long enough.  Add a component in a
    # leftmost eigenvector so that ||s|| = 2*lambda_lb/M.
    if lambda_lb > 0.0 && !singular_with_gradient &&
       y_pinv_norm <= r_lb * (1.0 + 10.0 * secular_tol)
        tau2 = max(r_lb^2 - y_pinv_norm^2, 0.0)
        y = copy(y_pinv)
        y[1] += sqrt(tau2)
        s = U * y
        return s, lambda_lb, true
    end

    # Regular case. Solve
    #
    #   ||(H + lambda I)^(-1) g|| = 2 lambda / M,
    #
    # for lambda > max(0,-lambda_min(H)).
    function secular(lambda)
        acc = 0.0
        @inbounds for i in 1:n
            den = d[i] + lambda
            if den <= 0.0
                return Inf
            end
            z = gt[i] / den
            acc += z * z
        end
        return sqrt(acc) - 2.0 * lambda / M
    end

    # Stay strictly inside the positive-definite domain.
    lo = lambda_lb
    if lambda_lb > 0.0
        lo = nextfloat(lambda_lb)
    end

    flo = secular(lo)
    if !(flo > 0.0)
        # This can occur only through finite-precision boundary effects.
        # Move a small, scale-aware distance into the interior.
        lo = lambda_lb + max(1e-14, 1e-12 * max(1.0, abs(lambda_lb)))
        flo = secular(lo)
    end

    # Find an upper bracket. The secular function is strictly decreasing in
    # the regular case and tends to -Inf as lambda -> Inf.
    hi = max(1.0, 2.0 * max(lo, 1e-12))
    fhi = secular(hi)
    bracket_it = 0
    while fhi > 0.0 && bracket_it < 100
        hi *= 2.0
        fhi = secular(hi)
        bracket_it += 1
    end
    fhi <= 0.0 || error("CR secular equation could not be bracketed.")

    lambda = 0.5 * (lo + hi)
    for _ in 1:max_secular
        lambda = 0.5 * (lo + hi)
        fmid = secular(lambda)

        if abs(fmid) <= secular_tol * max(1.0, 2.0 * lambda / M)
            break
        elseif fmid > 0.0
            lo = lambda
        else
            hi = lambda
        end

        if hi - lo <= secular_tol * max(1.0, abs(lambda))
            lambda = 0.5 * (lo + hi)
            break
        end
    end

    y = similar(gt)
    @inbounds for i in 1:n
        y[i] = -gt[i] / (d[i] + lambda)
    end
    s = U * y

    return s, lambda, false
end

function cr_optimize(
    nlp;
    initial_theta,
    max_iter = 1000,
    tol = 1e-5,
    M0 = 1.0,
    L0 = M0,
    max_search = 60,
    secular_tol = 1e-12,
    max_secular = 100,
)
    reset!(nlp)
    x = copy(initial_theta)
    M = Float64(M0)
    L_floor = Float64(L0)
    M > 0.0 || error("CR requires M0 > 0.")
    L_floor > 0.0 || error("CR requires L0 > 0.")
    start = time()

    hvp_c = 0
    nobj_c = 0
    ngrad_c = 0

    f = obj(nlp, x); nobj_c += 1
    g = grad(nlp, x); ngrad_c += 1
    g_norm = norm(g)

    local it = 0
    for k in 1:max_iter
        it = k
        g_norm < tol && break

        # Classical CR uses the full Hessian.  The existing benchmark problems
        # expose HVPs, so form H exactly from n basis-vector HVPs.
        H, nhvp = _cr_full_hessian(nlp, x)
        hvp_c += nhvp
        eig = eigen(Symmetric(H))

        accepted = false
        s = zeros(Float64, length(x))
        f_new = f

        # Nesterov--Polyak (2006), Section 5.2, strategy (5.7):
        # increase M until the trial decreases the objective.  The same
        # Hessian eigendecomposition is reused for every M trial.
        for _ in 1:max_search
            s, _, _ = _cr_cubic_global_step_from_eigen(
                eig, g, M;
                secular_tol=secular_tol,
                max_secular=max_secular,
            )

            snorm = norm(s)
            if snorm == 0.0
                accepted = true
                f_new = f
                break
            end

            x_trial = x .+ s
            f_trial = obj(nlp, x_trial); nobj_c += 1

            if f_trial <= f
                x = x_trial
                f_new = f_trial
                accepted = true
                break
            end

            M *= 2.0
        end

        accepted || error("CR line search failed to find an acceptable M.")
        f = f_new

        # Strategy (5.7): allow the next iteration to try a longer step.
        M = max(0.5 * M, L_floor)

        g = grad(nlp, x); ngrad_c += 1
        g_norm = norm(g)
    end

    return OptResult(
        "CR", time() - start, f, g_norm, it,
        hvp_c, nobj_c, ngrad_c, x,
    )
end

# ============================================================================
