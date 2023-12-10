function grad = gradientG(G,U,Usq, X)
% \mathcal{G} \times{ }_{1} \mathbf{U}_{1}^{\mathrm{T}} \mathbf{U}_{1} \cdots \times_{N} \mathbf{U}_{N}^{\mathrm{T}} \mathbf{U}_{N}
% -\mathcal{X} \times{ }_{1} \mathbf{U}_{1}^{\mathrm{T}} \cdots \times_{N} \mathbf{U}_{N}^{\mathrm{T}}
    XU = ModalProduct_All(X, U, 'compress');
    GUsq = ModalProduct_All(G,Usq,'decompress');
    grad = GUsq - XU;
end