function [Core, U, Rank] = HOSVD(X,tol,R)

N = ndims(X);
normXsqr = TensorNorm(X,'fro')^2;
eigsumthresh = tol.^2 * normXsqr / N;

if ~isempty(R)
    Rank = R;
else
    Rank = zeros(1, N);
end

U = cell(1, N); Y = X; 
for n = 1:N
    Yn = double(reshape(permute(Y, [n 1:n-1 n+1:N]),size(Y,n),[]));
    Z = Yn*Yn';
    
    [V,D] = eig(Z);
    [eigvec,pi] = sort(diag(D),'descend');
    if Rank(n) == 0
        eigsum = cumsum(eigvec,'reverse');
        Rank(n) = find(eigsum > eigsumthresh, 1, 'last');  
    end

    U{n} = V(:,pi(1:Rank(n)));
    Y = ModalProduct(Y, U{n}, n, 'compress');
end
Core = Y;

end