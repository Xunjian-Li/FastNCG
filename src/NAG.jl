# ============================================================================
# NAG
# ============================================================================
function nag_optimize(nlp; initial_theta, max_iter = 1000, tol = 1e-5,
                      L0 = 1e0, gamma = 0.5, eta = 1, max_ls = 30)
    reset!(nlp)
    x = copy(initial_theta)
    y = copy(x)
    A_prev = 0.0
    L = L0
    start = time()

    hvp_c = 0; nobj_c = 0; ngrad_c = 0

    f = obj(nlp, x); nobj_c += 1
    g = grad(nlp, x); ngrad_c += 1
    g_norm = norm(g)

    local it = 0
    for k in 1:max_iter
        it = k
        g_theta = grad(nlp, y); ngrad_c += 1
        g_at_theta = grad(nlp, x); ngrad_c += 1
        g_norm_theta = norm(g_at_theta)
        g_norm_theta < tol && break

        d = -g_theta
        dTg = dot(d, g_theta)
        f_y = obj(nlp, y); nobj_c += 1

        x_new = y
        accepted = false
        for ls in 1:max_ls
            step = 1.0 / L
            x_try = y .+ step .* d
            f_try = obj(nlp, x_try); nobj_c += 1
            if f_try <= f_y + eta * step * dTg
                x_new = x_try
                accepted = true
                break
            else
                L = L / gamma
            end
        end
        if !accepted
            step = 1.0 / L
            x_new = y .+ step .* d
            L = L * 2.0
        else
            L = L * gamma
        end
        L = max(L, 1e-12)

        a_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * A_prev^2))
        tau = (A_prev - 1.0) / a_new
        y_new = x_new .+ tau .* (x_new .- x)

        momentum = x_new .- x
        if dot(momentum, -g_at_theta) < 0.0
            A_prev = 0.0
            y_new = copy(x_new)
        else
            A_prev = a_new
        end

        x = x_new
        y = y_new
        f = obj(nlp, x); nobj_c += 1
        g = grad(nlp, x); ngrad_c += 1
        g_norm = norm(g)
    end

    return OptResult("NAG", time() - start, f, g_norm, it, hvp_c, nobj_c, ngrad_c, x)
end
