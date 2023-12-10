function X_proj = ModalProduct(X,U,mode,direction)
% Compute the modal product along with the mode of tensor
% \mathcal{X} \times_{n} {\mathbf{U}}_{n}

    X_unf = Unfold(X,mode);
    
    if strcmp(direction,'compress')
        X_mult = U'*X_unf;
    elseif strcmp(direction,'decompress')
        X_mult = U*X_unf;
    else
        error('Modal product direction must be either "compress" or "uncompress"');
    end
    
    sz = size(X);
    sz(mode) = size(X_mult,1);
    X_proj = Fold(X_mult,mode,sz);
      
end
    
function X_unf = Unfold(X,mode)
% Compute the unfolding (matricization) of a tensor along a specified mode
% (size(X,n)) x (size(X,1)*...*size(X,n-1)*size(X,n+1)*...*size(X,N))
    N = ndims(X);
    X_unf = reshape(permute(X, [mode 1:mode-1 mode+1:N]),size(X,mode),[]);
end

function X = Fold(X_unf,mode,sz)
% Inverse operation of matrizicing, i.e. reconstructs the multi-way array from it's matriziced version.
% sz is vector containing original dimensions
    N = length(sz);
    if mode == 1
        perm = 1:N;
    else
        perm = [2:mode 1 mode+1:N];
    end
    X = permute(reshape(X_unf,sz([mode 1:mode-1 mode+1:N])),perm);
end