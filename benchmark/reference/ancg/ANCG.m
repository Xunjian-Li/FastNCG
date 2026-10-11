function [x, out] = ANCG(fh, gradh, hessh, x0, opts)
% Adaptive regularized Newton-Conjugate Gradient (ANCG)
%
%   [x, out] = ANCG(fh, gradh, hessh, x0, opts)
%
% Inputs
%   fh     : @(x) f(x)
%   gradh  : @(x) grad f(x)
%   hessh  : @(x) Hessian ∇^2 f(x)
%   x0     : starting vector
%   opts   : struct with fields (defaults in parentheses)
%            .eps       (1e-6)   stationarity tolerance
%            .gamma0    (1)      initial regularization (≥ 1)
%            .zeta      (0.1)    CG relative accuracy in (0,1)
%            .eta       (0.25)   line-search parameter in (0, 1/2]
%            .theta     (0.5)    backtracking factor in (0,1)
%            .maxit     (1e5)    max iterations (safeguard)
%            .maxls     (200)    max backtracking steps
%            .verbose   (1)      0 quiet, 1 per-iter line
%            .gamma_cap (1e12)   safety cap for gamma
%
% Outputs
%   x   : final iterate
%   out : struct with fields:
%         .iters, .fevals, .gevals, .Hevals, .time
%         .fx_hist, .gn_hist, .gamma_hist, .alpha_hist, .dtype_hist (1=SOL,2=NC)
%         .stop (reason string)
%
% Notes
%   - Calls your capped-CG: [d, d_type] = capCG(H, g, eps_cg, zeta)
%     with d_type: 1 = SOL, 2 = NC.

% ---------- options ----------
tStart = cputime;
if nargin < 5, opts = struct(); end
get = @(s,d) iff(isfield(opts,s), opts.(s), d);

eps_tol  = get('eps',     1e-6);
gamma    = get('gamma0',  1);
zeta     = get('zeta',    0.1);
eta      = get('eta',     0.25);   % must be in (0, 1/2]
theta    = get('theta',   0.5);    % in (0,1)
maxit    = round(get('maxit',  1e5));
maxls    = round(get('maxls',   200));
verbose  = get('verbose', 1);
gammaCap = get('gamma_cap', 1e12);
warmstartCG = get('warmstartCG', 0);
beta = get('beta', 2);
restart = get('restart',0);

% validation
if ~(eta > 0 && eta <= 0.5)
    error('opts.eta must be in (0, 1/2].');
end
if ~(zeta > 0 && zeta < 1)
    error('opts.zeta must be in (0,1).');
end
if ~(theta > 0 && theta < 1)
    error('opts.theta must be in (0,1).');
end
gamma = max(1, gamma);  % enforce γ0 ≥ 1


% ---------- initialize ----------
x  = x0(:);
d=0;
fx = fh(x);  fe = 1;
gx = gradh(x); ge = 1;
Hx = hessh(x); He = 1;

fx_hist    = fx;
gn_hist    = norm(gx);
gamma_hist = gamma;
alpha_hist = [];
dtype_hist = [];  % 1=SOL, 2=NC

stop_reason = '';
k = 0;
hvp_count = 0;
if verbose
    fprintf('It  f(x)              ||g||           gamma       type    alpha      j   time  CGiter  hvp\n');
end


% ---------- main loop ----------
while norm(gx) > eps_tol && k < maxit
    k = k + 1;

    % CG damping epsilon: eps_cg = (gamma * ||g||)^(1/2)
    ng     = norm(gx);
    eps_cg = sqrt(gamma * ng);

    zeta = min(0.5,ng^(0.5));
    c_sol = eta*(1-eta)*theta/400;
    cmin = c_sol;

    % Capped-CG on (Hx + 2*eps_cg I) d = -gx
    if warmstartCG
        [d, d_type, dHd,cnt_tmp] = capCG(Hx, gx, eps_cg, zeta,d);   % 1=SOL, 2=NC
    else
        [d, d_type,dHd,cnt_tmp,inneriter] = capCG(Hx, gx, eps_cg, zeta);   % 1=SOL, 2=NC
    end
    hvp_count = hvp_count + cnt_tmp;
    if d_type == 2
        % ---------- NC branch ----------
        dg  = d' * gx;
        % dHd = d' * Hx * d;
        nd  = norm(d);
        if nd == 0 || ~isfinite(dHd)
            % Rare fallback
            dk  = -gx;
            ndk = norm(dk);
        else
            sgn = 1; if dg < 0, sgn = -1; end
            % d^k = -sgn(d^T g) * max{1, 1/gamma} * |d^T H d| / ||d||^3 * d
            dk  = - sgn * max(1, 1/gamma) * (abs(dHd) / max(nd^3, realmin)) * d;
            ndk = norm(dk);
        end

        % Backtracking:
        % f(x + theta^j dk) <= f(x) - (eta/2)*min{1,gamma}*theta^{2j}*||dk||^3
        j = 0; ok = false; f0 = fx; decBase = (eta/2)*min(1,gamma)*(ndk^3);
        while j <= maxls
            alpha   = theta^j;
            ftrial  = fh(x + alpha*dk); fe = fe+1;
            if isfinite(ftrial) && ftrial <= f0 - decBase*(theta^(2*j))
                ok = true; break;
            end
            j = j + 1;
        end
        if ~ok
            stop_reason = 'Line search failed (NC branch).';
            break;
        end

        % accept step
        alpha  = theta^j;                  % explicit
        x_new  = x  + alpha*dk;
        fx_new = ftrial;
        gx_new = gradh(x_new); ge = ge+1;
        Hx_new = hessh(x_new); He = He+1;
        dtype = 2;
        ng_new = norm(gx_new);
        if ng_new > 0.5*ng && alpha < theta/gamma
            gamma = min(beta*gamma, gammaCap);
        end

    else
        % ---------- SOL branch ----------
        dk  = d; ndk = norm(dk);

        % fast-accept: f(x+d) <= f(x) and ||g(x+d)|| <= ||g(x)||/2
        f1 = fh(x + dk); fe = fe+1;
        goto_accept = false;
        if isfinite(f1) && f1 <= fx
            g1 = gradh(x + dk); ge = ge+1;
            if norm(g1) <= 0.5*ng
                alpha  = 1; j = 0;
                x_new  = x + dk;
                fx_new = f1;
                gx_new = g1;
                Hx_new = hessh(x_new); He = He+1;
                goto_accept = true;
            end
        end

        if ~goto_accept
            % Standard Armijo:
            % f(x + theta^j d) <= f(x) - eta*(gamma*||g||)^{1/2}*theta^j*||d||^2
            j = 0; ok = false; f0 = fx;
            base = eta * sqrt(gamma * ng) * (ndk^2);
            while j <= maxls
                alpha  = theta^j;
                ftrial = fh(x + alpha*dk); fe = fe+1;
                if isfinite(ftrial) && ftrial <= f0 - base*(theta^j)
                    ok = true; break;
                end
                j = j + 1;
            end
            if ~ok
                stop_reason = 'Line search failed (SOL branch).';
                break;
            end

            % accept step (explicit alpha assignment)
            alpha  = theta^j;
            x_new  = x  + alpha*dk;
            fx_new = ftrial;
            gx_new = gradh(x_new); ge = ge+1;
            Hx_new = hessh(x_new); He = He+1;
        end
        dtype = 1;
        ng_new = norm(gx_new);
        if ng_new > 0.5*ng && (fx - fx_new) < cmin * gamma^(-1/2) * ng^(3/2)
            gamma = min(beta*gamma, gammaCap);
        end
    end

    % ng_new = norm(gx_new);
    % if ng_new > 0.5*ng && (fx - fx_new) < cmin * gamma^(-1/2) * ng^(3/2)
    %         gamma = min(beta*gamma, gammaCap);
    % end



    % commit
    x = x_new; fx = fx_new; gx = gx_new; Hx = Hx_new;
    % logs
    fx_hist(end+1,1)    = fx;                %#ok<AGROW>
    gn_hist(end+1,1)    = ng_new;          %#ok<AGROW>
    gamma_hist(end+1,1) = gamma;             %#ok<AGROW>
    alpha_hist(end+1,1) = alpha;             %#ok<AGROW>
    dtype_hist(end+1,1) = dtype;             %#ok<AGROW>

    if verbose && mod(k,100) == 0
        tchar = 'SOL'; if dtype==2, tchar='NC '; end
        fprintf('%-3d %-16.8e %-14.3e %-12.3e   %-3d  %-7.3e   %2d   %.2f  %d  %d\n', ...
            k, fx, gn_hist(end), gamma,  dtype,alpha, j, cputime-tStart, inneriter,cnt_tmp);
    end
    if mod(k,20) == 0 && restart
        gamma    = get('gamma0',  1);
    end
end

% ---------- termination ----------
if norm(gx) <= eps_tol
    stop_reason = 'First-order tolerance satisfied.';
elseif isempty(stop_reason)
    stop_reason = 'Max iterations reached.';
end

out.iters      = k;
out.fevals     = fe;
out.gevals     = ge;
out.Hevals     = He;
out.time       = cputime - tStart;
out.fx_hist    = fx_hist;
out.gn_hist    = gn_hist;
out.gamma_hist = gamma_hist;
out.alpha_hist = alpha_hist;
out.dtype_hist = dtype_hist;
out.stop       = stop_reason;
out.hvp_count = hvp_count;

end

% --------- tiny helper ---------
function y = iff(cond, a, b)
if cond, y = a; else, y = b; end
end
