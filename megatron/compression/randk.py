import torch
import random


def encoder(tensor, k):
    input_abs = torch.abs(tensor)
    input_abs_size = input_abs.size()
    input_abs_seq = torch.reshape(input_abs, (-1,))
    input_abs_seq_size = input_abs_seq.size()
    if k < input_abs_seq.size()[0]:
        indices = random.sample(range(0, input_abs_seq_size[0] - 1), k)
        indices = torch.tensor(indices).cuda()
        value = input_abs_seq[indices]
    else:
        indices = list(range(0, input_abs_seq_size[0] - 1))
        indices = torch.tensor(indices).cuda()
        value = input_abs_seq[indices]
    return value, indices, input_abs_size, input_abs_seq_size


def decoder(value, indices, input_abs_size, input_abs_seq_size):
    topk_sparse = torch.sparse_coo_tensor(indices.unsqueeze(0), value, input_abs_seq_size)
    topk_dense = topk_sparse.to_dense()
    res = torch.reshape(topk_dense, input_abs_size)
    return res
