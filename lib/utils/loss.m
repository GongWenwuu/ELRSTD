function obj = loss(X, Omega, G, USet, weight, Opts)
    Z = ModalProduct_All(G,USet,'decompress');
    if strcmp(Opts.init,'lrstd')
        obj = (Opts.beta/2)*TensorNorm(Omega.*(X-Z),'fro')^2 + TensorNorm(G,1) + nuclearnorm(USet,weight);
    else
        if Opts.alpha == 0
            obj = (Opts.beta/2)*TensorNorm(Omega.*(X-Z),'fro')^2 + nuclearnorm(USet,weight);
        else
            obj = (Opts.beta/2)*TensorNorm(Omega.*(X-Z),'fro')^2 + Opts.alpha*TensorNorm(G,1);
        end
    end        
end

function tnn = nuclearnorm(USet,weight)
tnn = 0; 
for n = 1:length(USet)
    tnn = tnn + weight(n)*TensorNorm(USet{n},2);
end
end