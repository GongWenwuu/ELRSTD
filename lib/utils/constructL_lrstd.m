function L = constructL_lrstd(X,neighbor_node,flag)
% Manifold is approximated by a linear combination of several graphs
% Guided by 'Multiple graph regularized nonnegative matrix factorization'

N = ndims(X); L = cell(1,N);
Num = length(neighbor_node);
WSet = cell(1,Num);
for n = 1: N
    Xmat = reshape(permute(X, [n 1:n-1 n+1:N]), size(X, n), []);
    for i = 1:Num
        WSet{i} = KernelWeight(Xmat,neighbor_node(i),[]);
    end

    L0 = cell(1,Num);
    DCol = cell(1,Num);
    DSet = cell(1,Num);
    for i = 1:Num
        sz = size(WSet{1},1);
        DCol{i} = full(sum(WSet{i},2));
        DSet{i} = spdiags(DCol{i},0,sz,sz);
        L0{i} = DSet{i} - WSet{i};
        if flag
            DSet{i} = spdiags(DCol{i}.^-.5,0,sz,sz);
            L0{i} = DSet{i}*L0{i}*DSet{i};
        end
    end

    tau = ones(1,Num)/Num;
    Ln = zeros(size(L0{1}));
    for i = 1:Num
        Ln = Ln+tau(i)*L0{i};
    end

    L{n} = Ln;
end

end

function W = KernelWeight(Network,KNN,options)
% Construct a weighted graph via KNN and then encoded by a symmetric affinity matrix using HeatKernel Weight Euclidean distance 
% Guided by Deng Cai, Xiaofei He and Jiawei Han, "Document Clustering Using Locality Preserving Indexing" IEEE TKDE, Dec. 2005.
    
    if ~isfield(options,'WeightMode')
        options.WeightMode = 'HeatKernel';
    end
    if ~isfield(options,'bSelfConnected')
        options.bSelfConnected = 1;
    end
    bBinary = 0;
    nodes = size(Network,1);
    maxM = 62500000; 
    BlockSize = floor(maxM/(nodes*3));
    G = zeros(nodes*(KNN+1),3);
    
    switch lower(options.WeightMode)
        case {lower('Binary')}
            bBinary = 1; 
        case {lower('HeatKernel')}
            if ~isfield(options,'t')
                nodes = size(Network,1);
                if nodes > 3000
                    D = EuDist2(Network(randsample(nodes,3000),:));
                else
                    D = EuDist2(Network);
                end
                options.t = mean(mean(D));
            end
    end
    
    for i = 1:ceil(nodes/BlockSize)
        if i == ceil(nodes/BlockSize)
            nodeIdx = (i-1)*BlockSize+1:nodes;
            dist = EuDist2(Network(nodeIdx,:),Network,0);

            nodesNow = length(nodeIdx);
            HKweight = zeros(nodesNow,KNN+1);
            idx = HKweight;
            for j = 1:KNN+1
                [HKweight(:,j),idx(:,j)] = min(dist,[],2);
                temp = (idx(:,j)-1)*nodesNow+(1:nodesNow)';
                dist(temp) = 1e100;
            end
            
            HKweight = exp(-HKweight/(2*options.t^2));

            G((i-1)*BlockSize*(KNN+1)+1:nodes*(KNN+1),1) = repmat(nodeIdx',[KNN+1,1]);
            G((i-1)*BlockSize*(KNN+1)+1:nodes*(KNN+1),2) = idx(:);
            G((i-1)*BlockSize*(KNN+1)+1:nodes*(KNN+1),3) = HKweight(:);
        else
            nodeIdx = (i-1)*BlockSize+1:i*BlockSize;
            dist = EuDist2(Network(nodeIdx,:),Network,0);

            nodesNow = length(nodeIdx);
            HKweight = zeros(nodesNow,KNN+1);
            idx = HKweight;
            for j = 1:KNN+1
                [HKweight(:,j),idx(:,j)] = min(dist,[],2);
                temp = (idx(:,j)-1)*nodesNow+(1:nodesNow)';
                dist(temp) = 1e100;
            end

            HKweight = exp(-HKweight/(2*options.t^2));

            G((i-1)*BlockSize*(KNN+1)+1:i*BlockSize*(KNN+1),1) = repmat(nodeIdx',[KNN+1,1]);
            G((i-1)*BlockSize*(KNN+1)+1:i*BlockSize*(KNN+1),2) = idx(:);
            G((i-1)*BlockSize*(KNN+1)+1:i*BlockSize*(KNN+1),3) = HKweight(:);
        end
    end

    W = sparse(G(:,1),G(:,2),G(:,3),nodes,nodes);
    
    if bBinary
        W(logical(W)) = 1;
    end
    
    if ~options.bSelfConnected
        W = W - diag(diag(W));
    end

    W = max(W,W');
    
end 

function D = EuDist2(M_a,M_b,bSqrt)

    if ~exist('bSqrt','var')
        bSqrt = 1;
    end

    if (~exist('fea_b','var')) || isempty(M_b)
        aa = sum(M_a.*M_a,2);
        ab = M_a*M_a';


        if issparse(aa)
            aa = full(aa);
        end

        D = bsxfun(@plus,aa,aa') - 2*ab;
        D(D<0) = 0;
        if bSqrt
            D = sqrt(D);
        end
        D = max(D,D');

    else
        aa = sum(M_a.*M_a,2);
        bb = sum(M_b.*M_b,2);
        ab = M_a*M_b';

        if issparse(aa)
            aa = full(aa);
            bb = full(bb);
        end

        D = bsxfun(@plus,aa,bb') - 2*ab;
        D(D<0) = 0;
        if bSqrt
            D = sqrt(D);
        end

    end
    
end