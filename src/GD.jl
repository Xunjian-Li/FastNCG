# ============================================================================
# GD
# ============================================================================
function gd_optimize(nlp; initial_theta, max_iter = 1000, tol = 1e-5,
                     L0 = 10000.0, gamma = 0.5, eta = 1, max_ls = 30)
    reset!(nlp)
    x = copy(initial_theta)
    L = L0
    start = time()

    hvp_c = 0; nobj_c = 0; ngrad_c = 0

    f = obj(nlp, x); nobj_c += 1
    g = grad(nlp, x); ngrad_c += 1
    g_norm = norm(g)

    local it = 0
    for k in 1:max_iter
        it = k
        g_norm < tol && break
        d = -g
        dTg = dot(d, g)

        accepted = false
        for ls in 1:max_ls
            step = 1.0 / L
            x_new = x .+ step .* d
            f_new = obj(nlp, x_new); nobj_c += 1
            if f_new <= f + eta * step * dTg
                x = x_new; f = f_new
                L = max(L * gamma, 1e-12)
                accepted = true
                break
            else
                L = L / gamma
            end
        end
        if !accepted
            step = 1.0 / L
            x = x .+ step .* d
            f = obj(nlp, x); nobj_c += 1
            L = max(L * 2.0, 1e-12)
        end

        g = grad(nlp, x); ngrad_c += 1
        g_norm = norm(g)
    end

    return OptResult("GD", time() - start, f, g_norm, it, hvp_c, nobj_c, ngrad_c, x)
end
