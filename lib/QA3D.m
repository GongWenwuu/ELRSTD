function [psnr, ssim] = QA3D(imagery1, imagery2)
% Evaluates the quality assessment indices for two tensors.
% Input:
%   imagery1 - the reference tensor
%   imagery2 - the target tensor


% Output:
%   psnr - Peak Signal-to-Noise Ratio
%   ssim - Structure SIMilarity
%
% by Yi Peng 'Kronecker-Basis-Representation Based Tensor Sparsity and Its Applications to Tensor Recovery'
% update by Wenwu Gong 18/04/2023

Nway = size(imagery1);
maxP = max(imagery1(:));

psnr_ = zeros(1,Nway(3));
ssim_ = zeros(1,Nway(3));
for i = 1:Nway(3)
    psnr_(i) = psnr_index(imagery1(:, :, i), imagery2(:, :, i), maxP);
    ssim_(i) = ssim_index(imagery1(:, :, i), imagery2(:, :, i));
end
psnr = mean(psnr_);
ssim = mean(ssim_);

end

function psnr = psnr_index(img1,img2,maxP)
    dim = numel(img1);
    psnr = 10*log10(dim*maxP^2/norm(img1(:)-img2(:))^2);
end

function [mssim, ssim_map] = ssim_index(img1, img2)

if (size(img1) ~= size(img2))
   mssim = -Inf;
   ssim_map = -Inf;
   return;
end

[M, N] = size(img1);

if ((M < 11) || (N < 11))
    mssim = -Inf;
    ssim_map = -Inf;
    return
end
window = fspecial('gaussian', 11, 1.5);
K = zeros(1,2);
K(1) = 0.01;
K(2) = 0.03;
L = max(img1(:));

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 && C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
   denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

return
end