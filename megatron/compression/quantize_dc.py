from collections import namedtuple

import torch
import numpy as np
from ..mpu.initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from torch.distributed import all_reduce, ReduceOp
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def _rounding(x, stochastic=False, minimum_stochastic_distance=0.2):
    if stochastic:
        x_floor = x.floor()
        th = x - x_floor
        if minimum_stochastic_distance > 0:
            th[th<minimum_stochastic_distance] = 0.
            th[th>1-minimum_stochastic_distance] = 1.
        pr = torch.rand_like(x)
        x_floor += (pr < th)
        return x_floor
    else:
        return x.round()

def _compress_nbits(x, bits, scale_method='max', scale_dims=(0,1)):

    fbits = bits - 1

    if scale_method == 'max':
        # issue: sensitive to outlier points
        scale = x.abs().amax(scale_dims, keepdims=True)
    elif scale_method == 'l2':
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean(scale_dims, keepdims=True).sqrt() * 2
    else:
        raise Exception('unkonwn scale method.')
    # fp16 should be enough
    scale = scale.half()
    x = x / (scale + 1e-6)

    x = x.ldexp(torch.tensor(fbits))
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1

    x = _rounding(x)
    x = x.clip(clip_min, clip_max)

    x = x - clip_min
    x = x.type(torch.uint8)

    return x, scale

def _decompress_nbits(x, scale, bits):

    fbits = bits - 1

    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1

    x = x.float() + clip_min

    x = x / (clip_max+1) * scale

    return x

def compress_8bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = _compress_nbits(x, bits=8, scale_method=scale_method, scale_dims=scale_dims)

    return x, scale


def decompress_8bit(x, scale):

    x = _decompress_nbits(x, scale, bits=8)

    return x

def compress_4bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = _compress_nbits(x, bits=4, scale_method=scale_method, scale_dims=scale_dims)

    x0, x1 = x.chunk(2, -1)
    x = (x0 << 4) + x1

    return x, scale


def decompress_4bit(x, scale):

    bitmask = 15

    x0 = (x >> 4)
    x1 = (x & bitmask)

    x = torch.cat([x0, x1], -1)

    x = _decompress_nbits(x, scale, bits=4)

    return x

def compress_2bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = _compress_nbits(x, bits=2, scale_method=scale_method, scale_dims=scale_dims)

    x0, x1, x2, x3 = x.chunk(4, -1)
    x = (x0 << 6) + (x1 << 4) + (x2 << 2) + x3

    return x, scale


def decompress_2bit(x, scale):

    bitmask = 3

    x0 = (x >> 6)
    x1 = (x >> 4) & bitmask
    x2 = (x >> 2) & bitmask
    x3 = x & bitmask
    x = torch.cat([x0, x1, x2, x3], -1)

    x = _decompress_nbits(x, scale, bits=2)

    return x



# Give up, no one cares too much accuracy
def quantize_accurate(x, num_bits=8):
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
