# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import sys
import time
import torch
import torch.nn.functional as F
import torch.nn.init as init
from scipy.linalg import hadamard
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import get_tensor_model_parallel_group
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import gather_multi_from_tensor_model_parallel_region
from .mappings import gather_from_sequence_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .mappings import reduce_scatter_to_sequence_parallel_region

from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from megatron import get_args, get_global_memory_buffer

from ..compression import quantize, topk, randk, srht, ct

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}

def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication and gradient accumulation
    fusion in backprop.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                async_grad_allreduce, sequence_parallel):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
      
        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        
        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(), async_op=True)

            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Convert the tensor shapes to 2D for execution compatibility
        # if use QR, please use reshape here
        grad_output = grad_output.reshape(grad_output.shape[0] * grad_output.shape[1],
                                          grad_output.shape[2])
        # grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
        #                                grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1],
				       total_input.shape[2])
 
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
 
        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input, 
                                                            group=get_tensor_model_parallel_group(),
                                                            async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # reduce scatter scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
        

        if ctx.gradient_accumulation_fusion:
            import fused_dense_cuda
            fused_dense_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.async_tensor_model_parallel_allreduce = (
                args.async_tensor_model_parallel_allreduce and
                world_size > 1)
        self.sequence_parallel = (
                args.sequence_parallel and
                world_size > 1)
        assert not self.async_tensor_model_parallel_allreduce or \
            not self.sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or \
                self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, bias, self.gradient_accumulation_fusion,
            self.async_tensor_model_parallel_allreduce, self.sequence_parallel)
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, layer_number,
                 bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.layer_number = layer_number
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.is_tensor_compress = args.is_tensor_compress
        self.tensor_compress_method = args.tensor_compress_method
        self.m = args.tensor_m
        self.k = args.tensor_k
        self.r = args.tensor_qr_r
        self.warmup_epoch = args.warmup_epoch
        self.warmup_iteration = args.warmup_iteration
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            if self.tensor_compress_method == 'ae':
                # torch.nn.init.xavier_uniform_ is used to avoid exploding gradient problem
                self.encoder = Parameter(torch.nn.init.xavier_uniform_(
                    torch.empty(args.tensor_ae_dim, self.output_size,
                                dtype=args.params_dtype))
                )
                self.decoder = Parameter(torch.nn.init.xavier_uniform_(
                    torch.empty(self.output_size, args.tensor_ae_dim,
                                dtype=args.params_dtype))
                )
            elif self.tensor_compress_method == 'srht':
                self.H_tensor = torch.tensor(hadamard(self.output_size), dtype=args.params_dtype)
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            if self.tensor_compress_method == 'ae':
                # torch.nn.init.xavier_uniform_ is used to avoid exploding gradient problem
                self.encoder = Parameter(torch.nn.init.xavier_uniform_(
                    torch.empty(args.tensor_ae_dim, self.output_size,
                                device=torch.cuda.current_device(), dtype=args.params_dtype)
                ))
                self.decoder = Parameter(torch.nn.init.xavier_uniform_(
                    torch.empty(self.output_size, args.tensor_ae_dim,
                                device=torch.cuda.current_device(), dtype=args.params_dtype)
                ))
            elif self.tensor_compress_method == 'srht':
                self.H_tensor = torch.tensor(hadamard(self.output_size),
                                             device=torch.cuda.current_device(), dtype=args.params_dtype)
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            setattr(self.bias, 'sequence_parallel', args.sequence_parallel)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.sequence_parallel = args.sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, None,
            self.gradient_accumulation_fusion, None, None)
        # All-reduce across all the partitions.
        args = get_args()
        if args.current_epoch < self.warmup_epoch:
            # use warmup technique here
            if self.sequence_parallel:
                output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
            else:
                output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            if self.is_tensor_compress and self.layer_number > 12:
                if self.tensor_compress_method == 'ae':
                    output_parallel = F.linear(output_parallel, self.encoder)
                    if self.sequence_parallel:
                        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
                    else:
                        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
                    output_ = F.linear(output_, self.decoder)

                elif self.tensor_compress_method == 'quantize':
                    output_parallel = output_parallel.to(torch.int16)
                    output_parallel = output_parallel.to(torch.int8)
                    # output_parallel, scale = quantize.compress_2bit(output_parallel)
                    if self.sequence_parallel:
                        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
                    else:
                        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
                    # output_ = quantize.decompress_2bit(output_, scale)
                    output_ = output_.to(torch.int16)
                    output_ = output_.to(torch.float16)

                elif self.tensor_compress_method == 'topk':
                    value, indices, input_abs_size, input_abs_seq_size = \
                        topk.encoder(output_parallel, k=self.k)
                    gather_list = gather_multi_from_tensor_model_parallel_region([value, indices])
                    world_size = get_tensor_model_parallel_world_size()
                    output_list = []
                    for i in range(0, world_size):
                        value = gather_list[0][i * self.k: (i + 1) * self.k]
                        indices = gather_list[1][i * self.k: (i + 1) * self.k]
                        output_list.append(topk.decoder(value, indices, input_abs_size, input_abs_seq_size))
                    output_ = output_list[0]
                    for i in range(len(output_list)):
                        if i != 0:
                            output_ = output_ + output_list[i].clone()

                elif self.tensor_compress_method == 'randk':
                    value, indices, input_abs_size, input_abs_seq_size = \
                        randk.encoder(output_parallel, k=self.k)
                    gather_list = gather_multi_from_tensor_model_parallel_region([value, indices])
                    world_size = get_tensor_model_parallel_world_size()
                    output_list = []
                    for i in range(0, world_size):
                        value = gather_list[0][i * self.k: (i + 1) * self.k]
                        indices = gather_list[1][i * self.k: (i + 1) * self.k]
                        output_list.append(randk.decoder(value, indices, input_abs_size, input_abs_seq_size))
                    output_ = output_list[0]
                    for i in range(len(output_list)):
                        if i != 0:
                            output_ = output_ + output_list[i].clone()

                elif self.tensor_compress_method == 'srht':
                    compress_tensor, S = srht.encoder(output_parallel, self.H_tensor,
                                                      d=self.output_size, m=self.m)
                    gather_list = gather_multi_from_tensor_model_parallel_region([compress_tensor, S])
                    world_size = get_tensor_model_parallel_world_size()
                    output_list = []
                    for i in range(0, world_size):
                        compress_tensor = gather_list[0][:, :, i * self.m: (i + 1) * self.m]
                        S = gather_list[1][:, i * self.output_size: (i + 1) * self.output_size]
                        output_list.append(srht.decoder(compress_tensor, S))
                    output_ = output_list[0]
                    for i in range(len(output_list)):
                        if i != 0:
                            output_ = output_ + output_list[i]

                elif self.tensor_compress_method == 'ct':
                    compress_tensor, S = ct.encoder(output_parallel, d=self.output_size, m=self.m)
                    gather_list = gather_multi_from_tensor_model_parallel_region([compress_tensor, S])
                    world_size = get_tensor_model_parallel_world_size()
                    output_list = []
                    for i in range(0, world_size):
                        compress_tensor = gather_list[0][:, :, i * self.m: (i + 1) * self.m]
                        S = gather_list[1][:, i * self.output_size: (i + 1) * self.output_size]
                        output_list.append(ct.decoder(compress_tensor, S))
                    output_ = output_list[0]
                    for i in range(len(output_list)):
                        if i != 0:
                            output_ = output_ + output_list[i]

                elif self.tensor_compress_method == 'qr':
                    Q = torch.randn(self.output_size,
                                    self.r, dtype=torch.float16).cuda()
                    P = torch.matmul(output_parallel, Q)
                    world_size = get_tensor_model_parallel_world_size()
                    if self.sequence_parallel:
                        P = reduce_scatter_to_sequence_parallel_region(P)
                    else:
                        P = reduce_from_tensor_model_parallel_region(P)
                    P = P.to(torch.float32)
                    P = P.permute(1, 0, 2)
                    P_hat = torch.linalg.qr(P).Q / world_size
                    P_hat = P_hat.to(torch.float16)
                    Q = torch.matmul(output_parallel.permute(1, 2, 0), P_hat)
                    if self.sequence_parallel:
                        Q = reduce_scatter_to_sequence_parallel_region(Q)
                    else:
                        Q = reduce_from_tensor_model_parallel_region(Q)
                    Q = Q / world_size
                    output_ = torch.matmul(P_hat, Q.permute(0, 2, 1))
                    output_ = output_.permute(1, 0, 2)

                elif self.tensor_compress_method == 'sign':
                    # we need to do reduce sign to 2 bits here
                    output_parallel = torch.sign(output_parallel)
                    if self.sequence_parallel:
                        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
                    else:
                        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
                    output_ = torch.sign(output_)

                else:
                    raise ValueError("Tensor Compression Method is Wrong")

            else:
                if self.sequence_parallel:
                    output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
                else:
                    output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

