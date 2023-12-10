function [X, rank, Sigma] = tracenorm(Z, tau)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       min: 1/2*||Z-X||^2 + tau ||X||_*
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \mathcal{D}_{\tau}(Z) = U \mathcal{D}_{\tau}(\Sigma) V',  \mathcal{D}_{\tau}(\Sigma)=max(diag(\sigma_{i})-\tau,0)

% [U,Sigma,V] = svd(Z);
% rank = sum(diag(Sigma) > tau);
% X = U(:,1:rank)*diag(Sigma(1:rank)-tau)*V(:,1:rank)';

    AAT = Z*Z';
    [U, Sigma2, ~] = svd(AAT);
    Sigma2 = diag(Sigma2);

    Sigma = sqrt(Sigma2); 
    tol = max(size(Z)) * eps(max(Sigma));
    rank = sum(Sigma > max(tol, tau));
    Sigmanew = max(Sigma(1:rank)-tau, 0) ./ Sigma(1:rank) ;
    X = U(:, 1:rank) * diag(Sigmanew) * U(:, 1:rank)'*Z;

end