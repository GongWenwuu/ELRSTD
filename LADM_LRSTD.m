function [Xest, U, hist] = LADM_LRSTD(X,Omega,Opts) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                          Proximal Linearized ALM for LRSTD                                        %
% \underset{\mathcal{G},\{\mathbf{U}_{n}\}}{\operatorname{min}} \ (1-\alpha) \prod_{n=1}^{N}\left\|\mathbf{U}_{n}\right\|_{*} +\alpha\|\mathcal{G}\|_{1}}
% \frac{\mu}{2} \left\|\mathcal{X}-\mathcal{G} \times_{n=1}^{N} \mathbf{U}_{n}\right\|_{\mathrm{F}}^{2} + \left\langle\mathcal{Y}, \mathcal{X}-\mathcal{G} \times_{n=1}^{N} \mathbf{U}_{n} \right\rangle
%                                       This code was written by Wenwu Gong (2023.03)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(Opts,'maxit'); maxit = Opts.maxit;  else; maxit = 300;   end
if isfield(Opts,'nu');    nu = Opts.nu;        else; nu = 1;        end
if isfield(Opts,'phi');   phi = Opts.phi;      else; phi = 0;       end
if isfield(Opts,'tol');   tol = Opts.tol;      else; tol = 1e-4;    end
if isfield(Opts, 'epsilon'); epsilon = Opts.epsilon; else; epsilon = 1e-4;     end 

N = ndims(X);
Z = X.*Omega;
figure('Position',get(0,'ScreenSize'));
subplot(1,3,1);ImageShow3D(Z(:,:,3));title('incomplete tensor'); 
Z(~Omega) = mean(X(Omega));
[Ginit,Uinit,~] = Initial(Z,Opts.Rpara,Opts.init);  
mu = 1e-5; 
mu_rho = 1.15;
sample_ra = sum(Omega(:))/numel(X);
if sample_ra < 0.05
    mu_rho = 1.1;
end
Y = zeros(size(Z));

L = cell(1,N); T = cell(1,N); gamma = zeros(1,N);
if isfield(Opts, 'flag')
    for n = 1:N
        Xmat = reshape(permute(X, [n 1:n-1 n+1:N]), size(X, n), []);
        if Opts.flag(n) == 1
            if strcmp(Opts.prior, 'stdc')
                L = constructL_stdc(size(X), {1, 2, 3}, 2, L);
            elseif strcmp(Opts.prior, 'lrstd')
                L = constructL_lrtd(X, 5, 0);
            else
                L{n} = eye(size(X, n));
            end
            gamma(n) = norm(Xmat, 2)/(2*norm(L{n}, 2));
        else
            gamma(n) = 0;
        end   
    end
else
    Opts.flag = 2.*ones(1,N);
end

Usq = cell(1,N); w_n = size(Z);
for n = 1:N
    Usq{n} = Uinit{n}'*Uinit{n};
    if ~isfield(Opts, 'weight')
        w_n(n) = Weight(X, n, 0,[]);
    else
        w_n(n) = Weight(X, n, Opts.alpha, Opts.weight);
    end
end
obj0 = loss(Z, Omega, Ginit, Uinit, w_n, Opts);

t0 = 1; delta = 1.1; niter = 0;
Gextra = Ginit; Uextra = Uinit; U = Uinit; 
Lgnew = 1; LU0 = ones(N,1); LUnew = ones(N,1);
gradU = cell(N,1); wU = ones(N,1);

% time = tic;
for iter = 1:maxit

    gradG = gradientG(Gextra, U, Usq, Z+Y/mu);
    Lg0 = Lgnew;
    Lgnew = lipG(Usq, delta);
    G = thresholding(Gextra - gradG/Lgnew, 1/(mu*Lgnew));
    for n = 1:N
        gradU{n} = gradientU(Uextra, U, Usq, sqrt(mu)*G, (mu*Z+Y)/sqrt(mu), U{n}, 0, L{n}, T{n}, 1, gamma(n), n, Opts.flag(n));
        LU0(n) = LUnew(n);
        LUnew(n) = lipU(Usq, sqrt(mu)*G, 0, L{n}, T{n}, 1, gamma(n), n, Opts.flag(n), delta);
        [U{n}, ~, ~] = tracenorm(Uextra{n} - gradU{n}/LUnew(n), w_n(n)/LUnew(n));   
        Usq{n} = U{n}'*U{n}; 
    end
    
    Z_pre = Z;
    Z_new = ModalProduct_All(G,U,'decompress');
    Z(~Omega) = Z_new(~Omega) - Y(~Omega)/mu;
    Z(Omega) = X(Omega) + phi*(Z(Omega) - Z_new(Omega));
    
    Y = Y + mu*nu*(Z - Z_new);
    mu = mu*mu_rho;
    
    % -- diagnostics and reporting --
    objk = loss(Z, Omega, G, U, w_n, Opts); 
    hist.obj(iter) = objk;
    relchange = norm(Z(:)-Z_pre(:))/norm(Z(:));
    hist.rel(1,iter) = relchange;
    relerr = abs(objk - obj0)/(obj0 + 1);
    hist.rel(2,iter) = relerr;
    rmse = sqrt((1/length(nonzeros(~Omega)))*norm(X(~Omega)-Z(~Omega),2)^2);
    hist.rmse(iter) = rmse;
    rse = norm(X(~Omega)-Z(~Omega))/norm(X(:));
    hist.rse(iter) = rse;
    nmae = norm(X(~Omega)-Z(~Omega),1)/norm(X(~Omega),1);
    hist.nmae(iter) = nmae;
      
    % if mod(iter,10)==0 
    %      disp(['LRSTD completed at ',int2str(iter),'-th iteration step within ',num2str(toc(time)),' seconds ']);
    %      fprintf('===================================\n');
    %      fprintf('Objective = %e\t, rel_DeltaX = %d\t,relerr = %d\t,NMAE = %d\n',objk, relchange,relerr,nmae);
    % end
    subplot(1,3,2);plot(hist.rse);title('# iterations vs. RSEs');
    subplot(1,3,3);ImageShow3D(Z(:,:,3));title('completed tensor');
    axes('position',[0,0,1,1],'visible','off');
    pause(0.1);
    
    % -- stopping checks and correction --
    if relerr < epsilon; niter = niter +1; else; niter = 0; end
    if relchange < tol || niter > 2
        break;
    end
    
    t = (1+sqrt(1+4*t0^2))/2;
    w = (t0-1)/t;
    wG = min([w,0.999*sqrt(Lg0/Lgnew)]);
    Gextra = G + wG*(G - Ginit);
    for i = 1:N
    wU(i) = min([w,0.9999*sqrt(LU0(i)/LUnew(i))]);
    Uextra{i} = U{i}+wU(i)*(U{i}-Uinit{i});
    end
    Ginit = G; Uinit = U; t0 = t; obj0 = objk;

end

Xest = Z;

end