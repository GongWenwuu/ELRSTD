function Modal = ModalProduct_All(G, Ucell, direction)
% Compute the modal chain product
    N = ndims(G);
    Modal = G;
    for i = 1:N
        Modal = ModalProduct(Modal, Ucell{i}, i, direction);
    end
end