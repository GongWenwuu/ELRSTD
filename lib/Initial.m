function [G0,U0,Rank] = Initial(X,para,init)
    
    load('s.mat','s');
    rng(s);
    N = ndims(X);
    Nway = size(X);
    U0 = cell(1,N);
    if length(para) == 1 
        [~, ~, Rank] = HOSVD(X,para,[]);
    else
        Rank = para;
    end
    if strcmp(init,'ntd')||strcmp(init,'sntd')
        for n = 1:N
            U0{n} = max(0,orth(rand(Nway(n),Rank(n))));
        end
        G0 = rand(Rank);
    elseif strcmp(init,'rtd')
        Xnorm = 0.5*TensorNorm(X,'fro');
        for n = 1:N
            U0{n} = randn(Nway(n),Rank(n));
            U0{n} = (U0{n}/norm(U0{n},'fro'))*Xnorm^(1/(N+1));
        end
        G0 = randn(Rank);
        G0 = (G0/TensorNorm(G0,'fro')*Xnorm^(1/(N+1)));
    elseif strcmp(init,'lrstd')
        for n = 1:N
            U0{n} = rand(Nway(n),Rank(n));
        end
        G0 = randn(Rank);
   elseif strcmp(init,'hosvd')
        if length(para) == 1 
            [G0, U0, Rank] = HOSVD(X,para,[]);
        else
            [G0, U0, Rank] = HOSVD(X,1,para);
        end
    else
        error('Initial parameter is wrong');
    end
    
end  