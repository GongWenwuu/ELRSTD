function A = ImageShow3D(I, scale)
% MSI/HSI image enhancement for visualization.
%
% Usage:
% display an image: ImageShow3D(I, 0.05)
% write an image: A = ImageShow3D(I, 0.05); imwrite(A, 'output.jpg')
%
% by Shuang Xu, TGRS, 2023.

if nargin==1
    scale=0.005;
end
I = double(I);

if ismatrix(I)
    q = quantile(I(:),[scale, 1-scale]);
    [low, high] = deal(q(1),q(2));
    I(I>high) = high;
    I(I<low) = low;
    I = (I-low)/(high-low);
else
    for i = 1:size(I,3)
        temp = I(:,:,i);
        q = quantile(temp(:),[scale, 1-scale]);
        [low, high] = deal(q(1),q(2));
        temp(temp>high) = high;
        temp(temp<low) = low;
        temp = (temp-low)/(high-low);
        I(:,:,i) = temp;
    end
end

if nargout == 1
    A = I;
else
    imshow(I)
end
end