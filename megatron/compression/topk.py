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


def encoder_feedback(tensor, k):
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
    topk_sparse = torch.sparse_coo_tensor(indices.unsqueeze(0), value, input_abs_seq_size)
    topk_dense = topk_sparse.to_dense()
    res = torch.reshape(topk_dense, input_abs_size)
    tensor = torch.reshape(tensor, input_abs_size)
    error_feedback = tensor - res
    return value, indices, input_abs_size, input_abs_seq_size, error_feedback

