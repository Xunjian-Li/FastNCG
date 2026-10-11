function [d,d_type,dHd,hvp_calls,iter] = capCG(H,g,e,zta,y_init)
% capCG  Capped conjugate-gradient subroutine used by HNCG and ANCG.
%
%   [d,d_type,dHd,hvp_calls,iter] = capCG(H,g,e,zta)
%   [d,d_type,dHd,hvp_calls,iter] = capCG(H,g,e,zta,y_init)
%
% Inputs:
%   H       : Hessian matrix at the current iterate.
%   g       : Gradient vector at the current iterate.
%   e       : Regularization parameter. The CG system is built with
%             H + 2*e*I.
%   zta     : Relative residual tolerance parameter used in the capped-CG
%             stopping rule.
%   y_init  : Optional warm-start vector. If omitted, the method starts from
%             the zero vector.
%
% Outputs:
%   d         : Returned direction.
%   d_type    : Direction type. d_type = 1 means an approximate Newton/CG
%               direction; d_type = 2 means a negative-curvature direction.
%   dHd       : Quadratic curvature value d'*H*d.
%   hvp_calls : Number of Hessian-vector products H*v used by this routine.
%   iter      : Number of CG iterations.
%
% Notes:
%   This routine applies CG to the regularized system
%
%       (H + 2*e*I)d = -g,
%
%   while monitoring curvature conditions. It either returns an approximate
%   solution direction or detects a negative-curvature direction.


% Initial upper estimate used in the capped-CG bounds.
U = 0;
[n,~] = size(H);

hvp_calls = 0;

    % Count one Hessian-vector product whenever H*v is evaluated.
    function y = Hv(v)
        hvp_calls = hvp_calls + 1;
        y = H * v;
    end
    function y = Hbar(v)
        y = Hv(v) + 2*e*v;
    end

H_bar = H + 2*e*eye(n);
kapa = (U+2*e)/e;
zta_ht = zta/(3*kapa);
tau = sqrt(kapa)/(sqrt(kapa)+1);
T = 4*kapa^4/(1-sqrt(tau))^2;


% Initialization: zero start by default, or warm start if y_init is supplied.
if nargin < 5
    y0 = 0; r0 = g; p0 = -g; j = 0;
else
    y0 = y_init;
    r0 = g - H_bar * y0;
    p0 = -r0;
end



Hb_p0 = Hbar(p0);        % counts 1 HVP
Hp0   = Hb_p0 - 2*e*p0;
if (p0)'* Hb_p0 < e * norm(p0,2)^2
    d = p0; d_type = 2;
elseif norm(Hp0,2) > U*norm(p0,2)
    U = norm(Hp0,2)/norm(p0,2);
    kapa = (U+2*e)/e;
    zta_ht = zta/(3*kapa);
    tau = sqrt(kapa)/(sqrt(kapa)+1);
    T = 4*kapa^4/(1-sqrt(tau))^2;
end

y_hty = zeros(n,n+1);
rj = r0; pj = p0; yj = y0; y_hty(:,1) = y0;

Hp_curr = Hv(pj);
iter = 0;
% Main capped-CG loop.
while 1
    iter = iter + 1;
    Hp_old  = Hp_curr;                 % 1 HVP/iter
    Hbp_old = Hp_old + 2*e*pj;
    aj = rj'*rj / (pj'*Hbp_old);
    yj = yj + aj * pj;
    rjp1 = rj + aj * Hbp_old;
    bj = norm(rjp1,2)^2/norm(rj,2)^2;
    rj = rjp1;
    pj = -rj + bj * pj;
    j = j+1;
    y_hty(:,j+1) = yj;

    Hp_curr = Hv(pj);
    if norm(Hp_curr,2) > U*norm(pj,2)
        U = norm(Hp_curr,2)/ norm(pj,2);
        kapa = (U+2*e)/e;
        zta_ht = zta/(3*kapa);
        tau = sqrt(kapa)/(sqrt(kapa)+1);
        T = 4*kapa^4/(1-sqrt(tau))^2;
    end


    Hyj = Hv(yj);
    if norm(Hyj,2) > U*norm(yj,2)
        U = norm(Hyj,2)/ norm(yj,2);
        kapa = (U+2*e)/e;
        zta_ht = zta/(3*kapa);
        tau = sqrt(kapa)/(sqrt(kapa)+1);
        T = 4*kapa^4/(1-sqrt(tau))^2;
    end

    Hrj = Hv(rj);
    if norm(Hrj,2) > U*norm(rj,2)
        U = norm(Hrj,2)/ norm(rj,2);
        kapa = (U+2*e)/e;
        zta_ht = zta/(3*kapa);
        tau = sqrt(kapa)/(sqrt(kapa)+1);
        T = 4*kapa^4/(1-sqrt(tau))^2;
    end
    % Stopping tests: return either negative curvature or an approximate solution.
    if yj'* (Hyj + 2*e*yj) < e * norm(yj,2)^2
        d = yj; d_type = 2; dHd = yj'*Hyj; break
    elseif norm(rj,2) <= zta_ht * norm(r0,2)
        d = yj; d_type = 1; dHd = yj'*Hyj; break
    else
        Hbp_curr = Hp_curr + 2*e*pj;
        if pj' * Hbp_curr < e * norm(pj,2)^2
            d = pj; d_type = 2; dHd = pj'*Hp_curr; break
        elseif norm(rj,2) > sqrt(T)*tau^(j/2) * norm(r0,2)
            aj = rj'*rj / (pj'*Hbp_curr);
            yjp1 = yj + aj * pj;
            for i = 0:j-1
                ydiff = yjp1 - y_hty(:,i+1);
                Hy    = Hv(ydiff);
                if (ydiff)'* (Hy + 2*e*ydiff) < e * norm(ydiff,2)^2
                    d = ydiff; d_type = 2; dHd = ydiff'*Hy;break
                end
            end
            break
        end
    end
end

end

function v = getf(s, name, def)
if isfield(s,name), v = s.(name); else, v = def; end
end
