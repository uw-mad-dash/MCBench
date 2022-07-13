from collections import namedtuple

import torch
import numpy as np
from ..mpu.initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from torch.distributed import all_reduce, ReduceOp

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])
def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    # To prevent over-flow in the SUM all-reduce afterwards, we need to multiply the scale by number of workers in the group
    group_size = get_tensor_model_parallel_world_size()
    all_reduce(max_val, group=get_tensor_model_parallel_group(), op=ReduceOp.MAX)
    all_reduce(min_val, group=get_tensor_model_parallel_group(), op=ReduceOp.MIN)
    
    min_val = min_val * group_size
    max_val = max_val * group_size
    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return q_x, scale, zero_point

def dequantize_tensor(q_tensor, q_scale, q_zero):
    return q_scale * (q_tensor.float() - q_zero)

"""
Quantization scheme for QSGD
Follows Alistarh, 2017 (https://arxiv.org/abs/1610.02132) but without the compression scheme.
"""
def quantize_qsgd(x, d):
    """quantize the tensor x in d level on the absolute value coef wise"""
    norm = np.sqrt(np.sum(np.square(x)))
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    ret = np.sign(x) * norm * new_level / d
    return ret
