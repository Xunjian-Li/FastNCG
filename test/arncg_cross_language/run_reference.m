function run_reference(root)
addpath(fullfile(fileparts(mfilename('fullpath')),'..','..','benchmark','reference','arncg'));
maxNumCompThreads(1);
names={'stationary','quadratic','rosenbrock','negative_curvature'};
for k=1:numel(names)
 name=names{k}; H=[4,1;1,2];
 if strcmp(name,'stationary') || strcmp(name,'quadratic')
  x0=[1;-1]; if strcmp(name,'stationary'), x0=[0;0]; end
  f=@(x) x'*H*x/2; g=@(x) H*x; hv=@(x,v) H*v;
 elseif strcmp(name,'rosenbrock')
  x0=[-1.2;1];
  f=@(x) 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
  g=@(x) [-400*x(1)*(x(2)-x(1)^2)+2*(x(1)-1);200*(x(2)-x(1)^2)];
  hv=@(x,v) [1200*x(1)^2-400*x(2)+2,-400*x(1);-400*x(1),200]*v;
 else
  x0=[0.1;1]; f=@(x) (x(1)^2-1)^2/4+x(2)^2/2;
  g=@(x) [x(1)^3-x(1);x(2)]; hv=@(x,v) [(3*x(1)^2-1)*v(1);v(2)];
 end
 calls=0;
 o=struct('max_iter',100000,'max_omega',inf,'beta',0.5,'mu',0.3,...
 'min_alpha',0.3,'cg_reltol',0.01,'cg_abstol',0.01,'cg_maxiter',4,...
 'gamma',5,'tau_minus',0.3,'tau_plus',1,'fallback_enabled',false,...
 'theta',1,'regularization_policy','gradient','acceleration_policy','gradient',...
 'cg_policy','recompute','minimal_norm_d',2e-16,'exit_for_many_unchanged_f_and_g',20,...
 'max_time',18000,'verbose',false,'dot_fn',@(x,u,v) sum(u.*v),...
 'norm_fn',@(x,v) norm(v),'loss_fn',f,'grad_fn',g,'hessvec_fn',@counted_hv);
 [x,gn,rec,~,ng,nf]=AdapNewtonCG(x0,1e-7,o);
 dlmwrite(fullfile(root,[name '_matlab.csv']),[numel(rec),nf,ng,calls,f(x),gn,x'],'precision','%.17g');
 hist=zeros(numel(rec),6);
 for j=1:numel(rec)
  r=rec(j); hist(j,:)=[r.f,r.norm_g,r.alpha,r.M,r.cg_it,1+strcmp(r.cg_dtype,'NC')];
 end
 dlmwrite(fullfile(root,[name '_matlab_history.csv']),hist,'precision','%.17g');
 fprintf('%s: iter=%d HVP=%d f=%.17g |g|=%.17g\n',name,numel(rec),calls,f(x),gn);
end
 function h=counted_hv(x,v)
  calls=calls+1; h=hv(x,v);
 end
end
