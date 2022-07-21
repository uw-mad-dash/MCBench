import torch


def encoder(tensor, k):
    input_abs = torch.abs(tensor)
    input_abs_size = input_abs.size()
    input_abs_seq = torch.reshape(input_abs, (-1,))
    input_abs_seq_size = input_abs_seq.size()
    rand_mat = torch.rand(input_abs_seq.size())
    if k < input_abs_seq.size()[0]:
        k_th_quant = torch.topk(rand_mat, k)[0][-1]
        mask = rand_mat >= k_th_quant
        indices = torch.nonzero(rand_mat >= k_th_quant)
        indices = torch.reshape(indices, (-1,))
        value = torch.masked_select(input_abs_seq, mask)
    else:
        k_th_quant = torch.topk(rand_mat, input_abs_seq.size()[0])[0][-1]
        mask = rand_mat >= k_th_quant
        indices = torch.nonzero(rand_mat >= k_th_quant)
        indices = torch.reshape(indices, (-1,))
        value = torch.masked_select(input_abs_seq, mask)
    return value, indices, input_abs_size, input_abs_seq_size


def decoder(value, indices, input_abs_size, input_abs_seq_size):
    topk_sparse = torch.sparse_coo_tensor(indices.unsqueeze(0), value, input_abs_seq_size)
    topk_dense = topk_sparse.to_dense()
    res = torch.reshape(topk_dense, input_abs_size)
    return res
