import numpy as np
import torch
import math


def encoder(input, H_tensor, d, m):
    D_tensor = torch.diag(torch.sign(torch.tensor(np.random.randint(0, 2, d)) - 1/2)).cuda()
    D_tensor = D_tensor.to(torch.float16)
    P_tensor = torch.tensor(np.random.randint(0, d, m)).cuda()
    S_tensor = torch.matmul(H_tensor[P_tensor, :], D_tensor) / math.sqrt(m)
    compress_tensor = torch.matmul(input, S_tensor.T)
    return compress_tensor, S_tensor


def decoder(input, S_tensor):
    return torch.matmul(input, S_tensor)
