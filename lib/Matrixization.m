function Mat_V = Matrixization(G, USet, mode, direction)
% Transfer Matrix Factorization problem by performing mode-k unfolding of Tucker decomposition 
% \|\mathcal{X}-\mathcal{G} \times_{n=1}^{N} {\mathbf{U}}_{n}\|_{F}^{2} 
% ==> \|{\mathbf{X}_{(n)}}-\mathbf{U}_{n} {\mathbf{G}_{(n)}} \mathbf{V}_{n}^{\mathrm{T}}\right\|_{\mathrm{F}}^{2}
% Mat_V = {\mathbf{G}_{(n)}} \mathbf{V}_{n}^{\mathrm{T}}, \mathbf{V}_{n} = \otimes_{p\neq n} \mathbf{U}_{p}
    modal = G; N = ndims(G);
    indices = 1:N;
    indices(mode) = [];
    for i = 1:numel(indices)
        modal = ModalProduct(modal,USet{indices(i)},indices(i), direction); 
    end
    Mat_V = reshape(permute(modal, [mode 1:mode-1 mode+1:N]),size(modal,mode),[]);
end