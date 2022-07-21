import torch


def encoder(tensor, k):
    input_abs = torch.abs(tensor)
    input_abs_size = input_abs.size()
    tensor = torch.reshape(tensor, (-1,))
    input_abs_seq = torch.reshape(input_abs, (-1,))
    input_abs_seq_size = input_abs_seq.size()
    if k < input_abs_seq.size()[0]:
        value, indices = torch.topk(input_abs_seq, k)
        value = tensor[indices]
    else:
        value, indices = torch.topk(input_abs_seq, input_abs_seq.size()[0])
        value = tensor[indices]
    return value, indices, input_abs_size, input_abs_seq_size


def decoder(value, indices, input_abs_size, input_abs_seq_size):
    topk_sparse = torch.sparse_coo_tensor(indices.unsqueeze(0), value, input_abs_seq_size)
    topk_dense = topk_sparse.to_dense()
    res = torch.reshape(topk_dense, input_abs_size)
    return res


# def topk(tensor, k):
#     input_abs = torch.abs(tensor)
#     input_abs_size = input_abs.size()
#     input_abs_seq = torch.reshape(input_abs, (-1,))
#     input_abs_seq_size = input_abs_seq.size()
#     if k < input_abs_seq.size()[0]:
#         value, indices = torch.topk(input_abs_seq, k)
#     else:
#         value, indices = torch.topk(input_abs_seq, input_abs_seq.size()[0])
#     topk_sparse = torch.sparse_coo_tensor(indices.unsqueeze(0), value, input_abs_seq_size)
#     topk_dense = topk_sparse.to_dense()
#     topk_dense_bool = torch.reshape(topk_dense, input_abs_size).bool()
#     res = torch.mul(tensor, topk_dense_bool)
#     return res
