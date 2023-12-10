function Fro = TensorNorm(X,form)

    n = ndims(X);
    Mat = reshape(shiftdim(X,n-1),size(X,n),[]);
    
    if (form == 2)
        Fro = norm(Mat,2);
    elseif (form == 0)
        Fro = norm(Mat,1); 
    elseif (form == 1)
        Fro = sum(sum(abs(Mat)));
    else
        Fro = norm(Mat,'fro');
    end
    
end