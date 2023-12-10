function L = lipG(U,delta)
% \| \mathbf{U}_{1}^{\mathrm{T}} {\mathbf{U}_{1}}\|_{\mathrm{F}} \| \mathbf{V}_{1}^{\mathrm{T}} {\mathbf{V}_{1}}\|_{\mathrm{F}}
    L = 1;
    for n = 1:length(U)
        L = delta*L*norm(U{n},2);
    end
end