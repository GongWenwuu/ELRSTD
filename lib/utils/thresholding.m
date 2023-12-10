function X = thresholding(Z, tau)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       min: 1/2*||Z-X||^2 + tau||X||_1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \mathcal{S}_{\tau}(Z) = sign(Z_{ij})(max{|Z_{ij}|-\tau,0});

    A_sign = sign(Z);
    X = abs(Z) - tau;
    X(X < 0) = 0;
    X = X .* A_sign;
    
end