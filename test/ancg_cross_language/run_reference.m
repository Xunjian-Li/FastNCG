function run_reference(root)
% Run the unmodified, pinned authors' functions on shared Julia-generated data.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'benchmark', 'reference', 'ancg'));
maxNumCompThreads(1);
names = {'quadratic','fail_sol','fail_nc','rosenbrock','polytope','tregression','logsumexp'};
opts = struct('eps',1e-7,'gamma0',1,'zeta',0.1,'eta',0.25,'theta',0.5, ...
    'maxit',10000,'maxls',200,'verbose',0,'gamma_cap',1e12, ...
    'warmstartCG',0,'beta',2,'restart',0);
% Every option is populated: the upstream get/iff helper eagerly reads fields.
for k = 1:numel(names)
    name = names{k}; folder = fullfile(root,name);
    A = readmatrix(fullfile(folder,'A.csv'));
    b = readmatrix(fullfile(folder,'b.csv'));
    x0 = readmatrix(fullfile(folder,'x0.csv'));
    [f,g,H] = callbacks(name,A,b);
    caseopts = opts;
    if startsWith(name,'fail_'), caseopts.maxls=0; end
    [x,out] = ANCG(f,g,H,x0,caseopts);
    writecsv(fullfile(folder,'matlab_x.csv'),x);
    writecsv(fullfile(folder,'matlab_history.csv'),[out.fx_hist,out.gn_hist,out.gamma_hist]);
    writecsv(fullfile(folder,'matlab_steps.csv'),[out.alpha_hist,out.dtype_hist]);
    writecsv(fullfile(folder,'matlab_counts.csv'),[out.iters;out.fevals;out.gevals;out.Hevals;out.hvp_count]);
    fid=fopen(fullfile(folder,'matlab_stop.txt'),'w'); fprintf(fid,'%s',out.stop); fclose(fid);
    fprintf('%s MATLAB: %d iterations, %s\n',name,out.iters,out.stop);
end
for h=[2,-4]
    [d,kind,dHd,calls,it]=capCG(h,1,1,0.1);
    writecsv(fullfile(root,sprintf('matlab_capcg_%d.csv',h)),[d,kind,dHd,calls,it]);
end
try
    capCG(2,1,1,0.1,0);
    outcome='unexpected success';
catch err
    outcome=err.message;
end
fid=fopen(fullfile(root,'matlab_warmstart.txt'),'w'); fprintf(fid,'%s',outcome); fclose(fid);
fid=fopen(fullfile(root,'matlab_version.txt'),'w'); fprintf(fid,'%s',version); fclose(fid);
end

function writecsv(path,M)
fid=fopen(path,'w');
for i=1:size(M,1)
    for j=1:size(M,2)
        if j>1, fprintf(fid,','); end
        fprintf(fid,'%.17g',M(i,j));
    end
    fprintf(fid,'\n');
end
fclose(fid);
end

function [f,g,H]=callbacks(name,A,b)
switch name
    case 'quadratic'
        f=@(x) sum(x.^2)/2; g=@(x) x; H=@(x) eye(numel(x));
    case {'fail_sol','fail_nc'}
        h0=0; if strcmp(name,'fail_nc'), h0=-4; end
        f=@(x) x(1)+h0*x(1)^2/2+1000*x(1)^4;
        g=@(x) 1+h0*x(1)+4000*x(1)^3;
        H=@(x) h0+12000*x(1)^2;
    case 'rosenbrock'
        f=@rosen_f; g=@rosen_g; H=@rosen_H;
    case 'polytope'
        r=@(x) max(A*x-b,0);
        f=@(x) sum(r(x).^4);
        g=@(x) A'*(4*r(x).^3);
        H=@(x) A'*((12*r(x).^2).*A);
    case 'tregression'
        r=@(x) A*x-b; nu=0.1;
        f=@(x) sum(log1p(r(x).^2/nu));
        g=@(x) A'*(2*r(x)./(nu+r(x).^2));
        H=@(x) A'*((2*(nu-r(x).^2)./(nu+r(x).^2).^2).*A);
    case 'logsumexp'
        f=@(x) lse_f(A*x-b); g=@(x) A'*lse_w(A*x-b);
        H=@(x) lse_H(A,x,b);
end
end
function value=rosen_f(x)
value=0;
for i=1:numel(x)-1
    r=x(i+1)-x(i)^2;
    value=value+100*r^2+(1-x(i))^2;
end
end
function result=rosen_g(x)
result=zeros(numel(x),1);
for i=1:numel(x)-1
    r=x(i+1)-x(i)^2;
    result(i)=result(i)+(-400*x(i)*r+2*(x(i)-1));
    result(i+1)=result(i+1)+200*r;
end
end
function result=rosen_H(x)
result=zeros(numel(x));
for i=1:numel(x)-1
    result(i,i)=result(i,i)+(1200*x(i)^2-400*x(i+1)+2);
    result(i,i+1)=result(i,i+1)-400*x(i);
    result(i+1,i)=result(i+1,i)-400*x(i);
    result(i+1,i+1)=result(i+1,i+1)+200;
end
end
function value=lse_f(z)
zmax=max(z); value=zmax+0.05*log(sum(exp((z-zmax)/0.05)));
end
function w=lse_w(z)
u=exp((z-max(z))/0.05); w=u/sum(u);
end
function H=lse_H(A,x,b)
w=lse_w(A*x-b); a=A'*w; H=(A'*(w.*A)-a*a')/0.05;
end
