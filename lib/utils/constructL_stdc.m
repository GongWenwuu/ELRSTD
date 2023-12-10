function L = constructL_stdc(tsize, VSet, gnns, Affinity)
% Guided by 'Simultaneous Tensor Decomposition and Completion Using Factor Priors'
% VSet: the subsets encoding sub-manifolds, e.g., {[1,2],[3]} means that
%       the 1st and 2nd sub-manifolds are encoded simultaneously and the
%       3rd sub-manifold is encoded individually
% Affinity: a cell array, where the i-th element denotes an affinity matrix
%           to encode the intra-factor relation w.r.t. the i-th tensor's
%           dimension
% gnns means the effective neighbors on manifold graphs
%% Initialization
num_g = numel(VSet);
N = numel(tsize);
id = cell(1,N);
for i = 1 : num_g
    id{i} = false(1,N);
    if sum(VSet{i}>N)>0
         error('Wrong index of manifold graph!');
    end
    id{i}(VSet{i}) = true;
end
VSet = id;

for i = 1 : N
    if isempty(Affinity{i}) || sum(abs(size(Affinity{i})-tsize(i)))~=0
        M = repmat(1:tsize(i),tsize(i),1);
        Affinity{i} = exp(-(M-M').^2/(2^2))-eye(tsize(i));
    end
end

% sub-graphs
A = numel(VSet);
% range of neighboring indeces 
gnns = -gnns:gnns;
L = cell(1,A); 

%% Main algorithm
for a = 1 : A

    % parameters in the m-th graph
    % # sub-manifolds
    mnum = sum(VSet{a});
    % possible neighborhood
    NSet = nchoosek(repmat(gnns,1,mnum),mnum);
    NSet(sum(abs(NSet),2)==0,:) = [];
    % # neighbors
    Nnum = size(NSet,1);
   
    % construction of the (downsampled) m-th graph
    aidx = 1:numel(tsize);
    aidx = aidx(VSet{a});
    Am = cell(1:mnum);
    for k = 1 : mnum
        Am{k} = imresize(Affinity{aidx(k)},round(tsize(aidx(k)))*ones(1,2));
    end
    ridx = ones(1,mnum);
    ridx(1) = 1;
    As = Am;
    %As{1} = Affinity{aidx(1)};
    msize = round(tsize(aidx).*ridx);
    N = prod(msize);

    % fill-in Laplacian matrix
    xidx = zeros(N*Nnum,1);
    yidx = xidx;
    zidx = xidx;
    len = 0;
    for j = 1 : N
        jidx = my_ind2sub(msize,j)';
        nidx = NSet+repmat(jidx,[Nnum,1]);
        nidx(sum(nidx<1,2)>0 | sum(nidx>repmat(msize,[Nnum,1]),2)>0,:) = [];
        nnum = size(nidx,1);
        edgeW = zeros([nnum,mnum]);
        for i = 1 : mnum
            edgeW(:,i) = As{i}((nidx(:,i)-1)*size(As{i},1)+jidx(i));
        end
        xidx(len+1:len+nnum) = j;
        yidx(len+1:len+nnum) = my_sub2ind(msize,nidx);
        zidx(len+1:len+nnum) = sum(edgeW,2);
        len = len+nnum;
    end
    D = sparse(xidx(1:len),yidx(1:len),zidx(1:len),N,N);
    D = (D+D')/2; 
    L{a} = full(diag(sum(D,2))-D);

    % % factorize Laplacian matrix
    % H{a} = full(cholcov(diag(sum(D,2))-D));
    % H{a} = reshape(H{a}',[msize,size(H{a},1)]);
end
end

% function R = cholcov(L)
% % CHOLCOV Cholesky-like decomposition for a covariance matrix.
% %   R = CHOLCOV(L) returns the Cholesky-like decomposition of the
% %   symmetric, positive definite matrix S, where L is a lower triangular
% %   matrix with the same size as S and L*L' = S. The output R is a
% %   upper triangular matrix such that R'*R = inv(L)*S*inv(L').
% 
% % Make sure S is symmetric
% if ~isequal(L, L')
%     error('BadL');
% end
% 
% % Compute the Cholesky-like decomposition of the covariance matrix
% p = size(L, 1);
% R = zeros(p, p);
% for j = 1:p
%     % Compute the jth element of the diagonal of R
%     R(j, j) = sqrt(max(0, L(j, j) - R(j, :)*R(j, :)'));
%     if R(j, j) == 0
%         error('BadL');
%     end
% 
%     % Compute the elements above the diagonal of R
%     for i = (j+1):p
%         R(i, j) = (L(i, j) - R(i, :)*R(j, :)') / R(j, j);
%     end
% end
% 
% % Make sure the diagonal elements of R are nonnegative
% if any(diag(R) < 0)
%     error('BadL');
% end
% 
% % Fill in the upper triangle of R
% R = triu(R)';
% end

function sub = my_ind2sub(siz,ndx)

siz = double(siz);
sub = zeros(numel(siz),numel(ndx));
k = [1 cumprod(siz(1:end-1))];
for i = numel(siz):-1:1
    vi = rem(ndx-1, k(i)) + 1;
    vj = (ndx - vi)/k(i) + 1;
    sub(i,:) = vj;
    ndx = vi;
end
end

function ndx = my_sub2ind(siz,sub)

siz = double(siz);
k = [1 cumprod(siz(1:end-1))];
ndx = 1;
for i = 1:numel(siz)
    v = sub(:,i);
    ndx = ndx + (v-1)*k(i);
end
end