using LinearAlgebra
using Random

export make_rosenbrock, make_logsumexp, make_polytope, make_tregression

# Compare values, not object identity: solvers reuse and mutate their x buffers.
# An explicit loop avoids allocating the broadcast temporary in all(a .== b).
@inline function _same_problem_point(a, b)
    axes(a) == axes(b) || return false
    @inbounds for i in eachindex(a, b)
        a[i] == b[i] || return false
    end
    return true
end

# Each factory owns mutable workspaces; use a separate instance per concurrent solve.

# ============================================================
# Rosenbrock
# ============================================================

function make_rosenbrock(dim::Int=200; x0=ones(dim) .* 2.0)
    x0 = Vector{Float64}(x0)

    function my_obj(x)
        total = 0.0
        @inbounds @simd for i in 1:(dim - 1)
            r = x[i + 1] - x[i]^2
            total += 100.0 * r^2 + (1.0 - x[i])^2
        end
        return total
    end

    function my_grad!(g, x)
        fill!(g, 0.0)

        @inbounds for i in 1:(dim - 1)
            xi = x[i]
            xip1 = x[i + 1]
            r = xip1 - xi^2

            g[i] += -400.0 * xi * r + 2.0 * (xi - 1.0)
            g[i + 1] += 200.0 * r
        end

        return g
    end

    function my_hprod!(Hv, x, v)
        fill!(Hv, 0.0)

        @inbounds for i in 1:(dim - 1)
            xi = x[i]
            xip1 = x[i + 1]

            h11 = 1200.0 * xi^2 - 400.0 * xip1 + 2.0
            h12 = -400.0 * xi

            Hv[i] += h11 * v[i] + h12 * v[i + 1]
            Hv[i + 1] += h12 * v[i] + 200.0 * v[i + 1]
        end

        return Hv
    end

    function my_objdiff(x, trial)
        total = 0.0
        correction = 0.0

        @inbounds for i in 1:(dim - 1)
            xi = x[i]
            xip1 = x[i + 1]
            yi = trial[i]
            yip1 = trial[i + 1]

            rx = xip1 - xi^2
            ry = yip1 - yi^2

            t = 100.0 * (ry - rx) * (ry + rx) +
                (xi - yi) * (2.0 - xi - yi)

            z = total + t
            correction += abs(total) >= abs(t) ? (total - z) + t : (t - z) + total
            total = z
        end

        return total + correction, nothing
    end

    return x0, my_obj, my_grad!, my_hprod!, my_objdiff
end


# ============================================================
# LogSumExp
# ============================================================

function make_logsumexp(dim::Int=200, m::Int=1000, rho::Float64=0.05;
                        seed::Int=42, x0=zeros(dim))
    rng = MersenneTwister(seed)

    A = randn(rng, m, dim) ./ sqrt(dim)
    b = randn(rng, m)
    x0 = Vector{Float64}(x0)

    # Latest objective/gradient point.
    linear = zeros(m)
    weights = zeros(m)
    obj_x = zeros(dim)
    obj_x_valid = Ref(false)
    obj_value = Ref(0.0)

    # Fixed outer/Hessian point.
    hess_linear = zeros(m)
    hess_w = zeros(m)
    hess_x = zeros(dim)
    hess_x_valid = Ref(false)

    # Trial relative to the fixed Hessian point.
    step = zeros(dim)
    trial_d = zeros(m)
    trial_x = zeros(dim)
    trial_base_x = zeros(dim)
    trial_d_valid = Ref(false)

    # HVP workspaces.
    Av = zeros(m)
    tmp = zeros(m)

    # objdiff fallback workspaces.
    diff_linear = zeros(m)
    diff_d = zeros(m)

    function update_weights!(w, z)
        zmax = maximum(z)

        total = 0.0
        @inbounds @simd for i in eachindex(z, w)
            wi = exp((z[i] - zmax) / rho)
            w[i] = wi
            total += wi
        end

        invtotal = 1.0 / total
        @inbounds @simd for i in eachindex(w)
            w[i] *= invtotal
        end

        return zmax + rho * log(total)
    end

    function my_obj(x)
        if !obj_x_valid[] || !_same_problem_point(x, obj_x)
            if hess_x_valid[] && !_same_problem_point(x, hess_x)
                @inbounds @simd for i in eachindex(step)
                    step[i] = x[i] - hess_x[i]
                end

                mul!(trial_d, A, step)

                @inbounds @simd for i in eachindex(linear, hess_linear, trial_d)
                    linear[i] = hess_linear[i] + trial_d[i]
                end

                copyto!(trial_x, x)
                copyto!(trial_base_x, hess_x)
                trial_d_valid[] = true
            elseif hess_x_valid[]
                copyto!(linear, hess_linear)
                trial_d_valid[] = false
            else
                mul!(linear, A, x)

                @inbounds @simd for i in eachindex(linear, b)
                    linear[i] -= b[i]
                end

                trial_d_valid[] = false
            end

            obj_value[] = update_weights!(weights, linear)
            copyto!(obj_x, x)
            obj_x_valid[] = true
        end

        return obj_value[]
    end

    function my_grad!(g, x)
        if !obj_x_valid[] || !_same_problem_point(x, obj_x)
            my_obj(x)
        end

        mul!(g, A', weights)
        return g
    end

    function my_hprod!(Hv, x, v)
        if !hess_x_valid[] || !_same_problem_point(x, hess_x)
            if obj_x_valid[] && _same_problem_point(x, obj_x)
                copyto!(hess_linear, linear)
                copyto!(hess_w, weights)
            else
                mul!(hess_linear, A, x)

                @inbounds @simd for i in eachindex(hess_linear, b)
                    hess_linear[i] -= b[i]
                end
                update_weights!(hess_w, hess_linear)
            end

            copyto!(hess_x, x)
            hess_x_valid[] = true
            trial_d_valid[] = false
        end

        mul!(Av, A, v)
        c = dot(hess_w, Av)

        @inbounds @simd for i in eachindex(tmp, hess_w, Av)
            tmp[i] = hess_w[i] * (Av[i] - c) / rho
        end

        mul!(Hv, A', tmp)
        return Hv
    end

    function my_objdiff(x, trial)
        if hess_x_valid[] && _same_problem_point(x, hess_x) &&
           trial_d_valid[] && _same_problem_point(trial, trial_x) &&
           _same_problem_point(x, trial_base_x)

            w = hess_w
            d = trial_d

            shift = -Inf
            @inbounds for i in eachindex(w, d)
                wi = w[i]
                wi <= 0.0 && continue
                t = log(wi) + d[i] / rho
                t > shift && (shift = t)
            end

            !isfinite(shift) && return (NaN, Inf)

            total = 0.0
            @inbounds for i in eachindex(w, d)
                wi = w[i]
                wi <= 0.0 && continue
                total += exp(log(wi) + d[i] / rho - shift)
            end

            delta = rho * (shift + log(total))
            return delta, nothing
        end

        mul!(diff_linear, A, x)

        @inbounds @simd for i in eachindex(diff_linear, b)
            diff_linear[i] -= b[i]
        end

        @inbounds @simd for i in eachindex(step)
            step[i] = trial[i] - x[i]
        end

        mul!(diff_d, A, step)

        zmax = maximum(diff_linear)

        denom = 0.0
        @inbounds for i in eachindex(diff_linear)
            denom += exp((diff_linear[i] - zmax) / rho)
        end

        shift = -Inf
        @inbounds for i in eachindex(diff_linear, diff_d)
            t = (diff_linear[i] - zmax + diff_d[i]) / rho
            t > shift && (shift = t)
        end

        numer = 0.0
        @inbounds for i in eachindex(diff_linear, diff_d)
            numer += exp(
                (diff_linear[i] - zmax + diff_d[i]) / rho - shift
            )
        end

        delta = rho * (shift + log(numer) - log(denom))
        return delta, nothing
    end

    return x0, my_obj, my_grad!, my_hprod!, my_objdiff
end


# ============================================================
# Polytope
# ============================================================

function make_polytope(dim::Int=400, m::Int=1000, p::Int=3;
                       seed::Int=42, x0=ones(dim))
    rng = MersenneTwister(seed)

    A = randn(rng, m, dim) ./ sqrt(dim)
    b = randn(rng, m)
    x0 = Vector{Float64}(x0)

    # Latest objective/gradient point.
    res = zeros(m)
    obj_x = zeros(dim)
    obj_x_valid = Ref(false)
    obj_value = Ref(0.0)
    obj_value_valid = Ref(false)

    # Fixed outer/Hessian point.
    hess_res = zeros(m)
    hess_coeff = zeros(m)
    hess_x = zeros(dim)
    hess_x_valid = Ref(false)

    # Trial relative to fixed Hessian point.
    step = zeros(dim)
    trial_d = zeros(m)
    trial_x = zeros(dim)
    trial_base_x = zeros(dim)
    trial_d_valid = Ref(false)

    # Gradient/HVP workspaces.
    Av = zeros(m)
    tmp = zeros(m)

    # objdiff fallback.
    diff_r = zeros(m)
    diff_d = zeros(m)

    function ensure_residual!(x)
        if !obj_x_valid[] || !_same_problem_point(x, obj_x)
            if hess_x_valid[] && !_same_problem_point(x, hess_x)
                @inbounds @simd for i in eachindex(step)
                    step[i] = x[i] - hess_x[i]
                end

                mul!(trial_d, A, step)

                @inbounds @simd for i in eachindex(res, hess_res, trial_d)
                    res[i] = hess_res[i] + trial_d[i]
                end

                copyto!(trial_x, x)
                copyto!(trial_base_x, hess_x)
                trial_d_valid[] = true
            elseif hess_x_valid[]
                copyto!(res, hess_res)
                trial_d_valid[] = false
            else
                mul!(res, A, x)

                @inbounds @simd for i in eachindex(res, b)
                    res[i] -= b[i]
                end

                trial_d_valid[] = false
            end

            copyto!(obj_x, x)
            obj_x_valid[] = true
            obj_value_valid[] = false
        end
        return nothing
    end

    function my_obj(x)
        ensure_residual!(x)
        obj_value_valid[] && return obj_value[]
        total = 0.0

        @inbounds for i in eachindex(res)
            ri = res[i]
            ri > 0.0 && (total += ri^p)
        end

        obj_value[] = total
        obj_value_valid[] = true
        return total
    end

    function my_grad!(g, x)
        ensure_residual!(x)

        @inbounds for i in eachindex(res, tmp)
            ri = res[i]
            tmp[i] = ri > 0.0 ? p * ri^(p - 1) : 0.0
        end

        mul!(g, A', tmp)
        return g
    end

    function my_hprod!(Hv, x, v)
        if !hess_x_valid[] || !_same_problem_point(x, hess_x)
            if obj_x_valid[] && _same_problem_point(x, obj_x)
                copyto!(hess_res, res)
            else
                mul!(hess_res, A, x)

                @inbounds @simd for i in eachindex(hess_res, b)
                    hess_res[i] -= b[i]
                end
            end

            @inbounds for i in eachindex(hess_res, hess_coeff)
                ri = hess_res[i]

                hess_coeff[i] =
                    ri > 0.0 ?
                    p * (p - 1) * ri^(p - 2) :
                    0.0
            end

            copyto!(hess_x, x)
            hess_x_valid[] = true
            trial_d_valid[] = false
        end

        mul!(Av, A, v)

        @inbounds @simd for i in eachindex(tmp, hess_coeff, Av)
            tmp[i] = hess_coeff[i] * Av[i]
        end

        mul!(Hv, A', tmp)
        return Hv
    end

    function my_objdiff(x, trial)
        if hess_x_valid[] && _same_problem_point(x, hess_x) &&
           trial_d_valid[] && _same_problem_point(trial, trial_x) &&
           _same_problem_point(x, trial_base_x)

            r = hess_res
            d = trial_d
        else
            mul!(diff_r, A, x)

            @inbounds @simd for i in eachindex(diff_r, b)
                diff_r[i] -= b[i]
            end

            @inbounds @simd for i in eachindex(step)
                step[i] = trial[i] - x[i]
            end

            mul!(diff_d, A, step)

            r = diff_r
            d = diff_d
        end

        total = 0.0
        correction = 0.0

        @inbounds for i in eachindex(r, d)
            a = max(r[i], 0.0)
            c = max(r[i] + d[i], 0.0)

            if p == 2
                t = (c - a) * (c + a)
            elseif p == 3
                t = (c - a) * (c^2 + c * a + a^2)
            elseif p == 4
                t = (c - a) * (c + a) * (c^2 + a^2)
            else
                t = c^p - a^p
            end

            z = total + t
            correction += abs(total) >= abs(t) ? (total - z) + t : (t - z) + total
            total = z
        end

        return total + correction, nothing
    end

    return x0, my_obj, my_grad!, my_hprod!, my_objdiff
end


# ============================================================
# Student-t regression
# ============================================================

function make_tregression(dim::Int=200, m::Int=500, nu::Float64=0.001;
                          seed::Int=42)
    rng = MersenneTwister(seed)

    x_true = randn(rng, dim)

    A = randn(rng, m, dim)

    for j in 2:dim
        A[:, j] .+= 0.3 .* A[:, j-1]
    end

    A ./= sqrt(1.0 + 0.3^2)

    noise = 0.5 .* randn(rng, m)
    chi = 2.0 .* rand(rng, m)
    b = A * x_true + noise ./ sqrt.(chi)

    x0 = A \ b

    # Latest objective/gradient point.
    res = zeros(m)
    obj_x = zeros(dim)
    obj_x_valid = Ref(false)
    obj_value = Ref(0.0)
    obj_value_valid = Ref(false)

    # Fixed outer/Hessian point.
    hess_res = zeros(m)
    hess_coeff = zeros(m)
    hess_x = zeros(dim)
    hess_x_valid = Ref(false)

    # Trial relative to fixed Hessian point.
    step = zeros(dim)
    trial_d = zeros(m)
    trial_x = zeros(dim)
    trial_base_x = zeros(dim)
    trial_d_valid = Ref(false)

    # Gradient/HVP workspaces.
    Av = zeros(m)
    tmp = zeros(m)

    # objdiff fallback.
    diff_r = zeros(m)
    diff_d = zeros(m)

    function ensure_residual!(x)
        if !obj_x_valid[] || !_same_problem_point(x, obj_x)
            if hess_x_valid[] && !_same_problem_point(x, hess_x)
                @inbounds @simd for i in eachindex(step)
                    step[i] = x[i] - hess_x[i]
                end

                mul!(trial_d, A, step)

                @inbounds @simd for i in eachindex(res, hess_res, trial_d)
                    res[i] = hess_res[i] + trial_d[i]
                end

                copyto!(trial_x, x)
                copyto!(trial_base_x, hess_x)
                trial_d_valid[] = true
            elseif hess_x_valid[]
                copyto!(res, hess_res)
                trial_d_valid[] = false
            else
                mul!(res, A, x)

                @inbounds @simd for i in eachindex(res, b)
                    res[i] -= b[i]
                end

                trial_d_valid[] = false
            end

            copyto!(obj_x, x)
            obj_x_valid[] = true
            obj_value_valid[] = false
        end
        return nothing
    end

    function my_obj(x)
        ensure_residual!(x)
        obj_value_valid[] && return obj_value[]
        total = 0.0

        @inbounds @simd for i in eachindex(res)
            total += log1p(res[i]^2 / nu)
        end

        obj_value[] = total
        obj_value_valid[] = true
        return total
    end

    function my_grad!(g, x)
        ensure_residual!(x)

        @inbounds @simd for i in eachindex(res, tmp)
            ri = res[i]
            tmp[i] = 2.0 * ri / (nu + ri^2)
        end

        mul!(g, A', tmp)
        return g
    end

    function my_hprod!(Hv, x, v)
        if !hess_x_valid[] || !_same_problem_point(x, hess_x)
            if obj_x_valid[] && _same_problem_point(x, obj_x)
                copyto!(hess_res, res)
            else
                mul!(hess_res, A, x)

                @inbounds @simd for i in eachindex(hess_res, b)
                    hess_res[i] -= b[i]
                end
            end

            @inbounds @simd for i in eachindex(hess_res, hess_coeff)
                ri = hess_res[i]
                den = nu + ri^2
                hess_coeff[i] = 2.0 * (nu - ri^2) / den^2
            end

            copyto!(hess_x, x)
            hess_x_valid[] = true
            trial_d_valid[] = false
        end

        mul!(Av, A, v)

        @inbounds @simd for i in eachindex(tmp, hess_coeff, Av)
            tmp[i] = hess_coeff[i] * Av[i]
        end

        mul!(Hv, A', tmp)
        return Hv
    end

    function my_objdiff(x, trial)
        if hess_x_valid[] && _same_problem_point(x, hess_x) &&
           trial_d_valid[] && _same_problem_point(trial, trial_x) &&
           _same_problem_point(x, trial_base_x)

            r = hess_res
            d = trial_d
        else
            mul!(diff_r, A, x)

            @inbounds @simd for i in eachindex(diff_r, b)
                diff_r[i] -= b[i]
            end

            @inbounds @simd for i in eachindex(step)
                step[i] = trial[i] - x[i]
            end

            mul!(diff_d, A, step)

            r = diff_r
            d = diff_d
        end

        total = 0.0
        correction = 0.0

        @inbounds for i in eachindex(r, d)
            ri = r[i]
            di = d[i]

            q = (2.0 * ri * di + di^2) / (nu + ri^2)

            (!isfinite(q) || q <= -1.0) &&
                return (NaN, Inf)

            t = log1p(q)

            z = total + t
            correction +=
                abs(total) >= abs(t) ?
                (total - z) + t :
                (t - z) + total

            total = z
        end

        return total + correction, nothing
    end

    return x0, my_obj, my_grad!, my_hprod!, my_objdiff
end