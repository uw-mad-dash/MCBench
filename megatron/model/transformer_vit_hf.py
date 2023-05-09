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

"""Transformer."""
import math
from contextlib import nullcontext
import sys
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard
from torch import nn
from torch.nn.parameter import Parameter
from queue import Queue

from megatron import get_timers, get_args, get_global_memory_buffer
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
from transformers.activations import ACT2FN

from ..compression import quantize, topk, randk, srht, ct

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + \
                        torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method, layer_number):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = ACT2FN["gelu"]
        # self.activation_func = F.gelu
        # if args.openai_gelu:
        #     self.activation_func = openai_gelu
        # elif args.onnx_safe:
        #     self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            layer_number,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = \
            self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class SwitchMLP(MegatronModule):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(self, init_method, output_layer_init_method, layer_number):
        super(SwitchMLP, self).__init__()
        args = get_args()
        self.router = torch.nn.Linear(args.hidden_size, args.num_experts)
        self.experts = torch.nn.ModuleList()
        for i in range(args.num_experts):
            self.experts.append(ParallelMLP(init_method, output_layer_init_method, layer_number))

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2)  # [s b 1]

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2))  # [s*b h]
        max_prob = max_prob.view(-1, max_prob.size(2))  # [s*b 1]
        max_ind = max_ind.view(-1)  # [s*b]

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)
        # TODO (rprenger) This does each expert in serial, but it could be parallelized

        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices, :]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices, :] = output
            output_bias_total[local_indices, :] = output_bias

        output_total = output_total * max_prob
        output_bias_total = output_bias_total * max_prob
        output_total = output_total.view(s, b, h)
        output_bias_total = output_bias_total.view(s, b, h)

        return output_total, output_bias_total


class CoreAttention(MegatronModule):

    def __init__(self, layer_number,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size / self.tensor_parallel_size)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        query_layer = query_layer.permute(0, 2, 1, 3).contiguous()
        key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
        value_layer = value_layer.permute(0, 2, 1, 3).contiguous()

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method)

        self.core_attention = CoreAttention(self.layer_number,
                                            self.attn_mask_type)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            layer_number,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        hidden_states = mpu.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
            batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
            batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                        :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                          :sequence_end, batch_start:batch_end, ...]

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(init_method, output_layer_init_method, layer_number)
        else:
            self.mlp = ParallelMLP(init_method, output_layer_init_method, layer_number)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
            nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        return output


class ParallelTransformerEncoderLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(ParallelTransformerEncoderLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.is_pipeline_compress = args.is_pipeline_compress
        self.pipeline_compress_method = args.pipeline_compress_method
        self.pipeline_ae_dim = args.pipeline_ae_dim
        self.pipeline_qr_r = args.pipeline_qr_r
        self.pipeline_k = args.pipeline_k

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(init_method, output_layer_init_method, layer_number)
        else:
            self.mlp = ParallelMLP(init_method, output_layer_init_method, layer_number)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
            nullcontext if use_nvfuser else torch.enable_grad

        if args.use_cpu_initialization:
            if self.is_pipeline_compress:
                if self.pipeline_compress_method == 'ae':
                    # torch.nn.init.xavier_uniform_ is used to avoid exploding gradient problem
                    self.encoder = Parameter(torch.nn.init.xavier_uniform_(
                        torch.empty(args.pipeline_ae_dim, args.hidden_size,
                                    dtype=args.params_dtype)
                    ))
                elif self.pipeline_compress_method == 'power':
                    self.generator = torch.Generator().manual_seed(0)
                    self.q_queue = Queue()
                    self.q_queue.put(
                        torch.randn(
                            [args.micro_batch_size, args.hidden_size, args.pipeline_qr_r],
                            generator=self.generator,
                            dtype=args.params_dtype
                        )
                    )
                elif self.pipeline_compress_method == 'ef_power':
                    self.generator = torch.Generator().manual_seed(0)
                    self.q_queue = Queue()
                    self.q_queue.put(
                        torch.randn(
                            [args.micro_batch_size, args.hidden_size, args.pipeline_qr_r],
                            generator=self.generator,
                            dtype=args.params_dtype
                        )
                    )
                    self.error_feedback = torch.zeros(
                        args.seq_length, args.micro_batch_size, args.hidden_size,
                        dtype=args.params_dtype
                    )
                elif self.pipeline_compress_method == 'topk':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   dtype=torch.int64)
                elif self.pipeline_compress_method == 'randk':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   dtype=torch.int64)
                elif self.pipeline_compress_method == 'srht':
                    self.H_tensor = torch.tensor(hadamard(args.hidden_size), dtype=torch.float16)
                elif self.pipeline_compress_method == 'topk_feedback':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   dtype=torch.int64)
                    self.error_feedback = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                      dtype=args.params_dtype)
                elif self.pipeline_compress_method == 'randk_feedback':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   dtype=torch.int64)
                    self.error_feedback = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                      dtype=args.params_dtype)
        else:
            if self.is_pipeline_compress:
                if self.pipeline_compress_method == 'ae':
                    # torch.nn.init.xavier_uniform_ is used to avoid exploding gradient problem
                    self.encoder = Parameter(torch.nn.init.xavier_uniform_(
                        torch.empty(args.pipeline_ae_dim, args.hidden_size,
                                    device=torch.cuda.current_device(), dtype=args.params_dtype)
                    ))
                elif self.pipeline_compress_method == 'power':
                    self.generator = torch.Generator(device=torch.cuda.current_device()).manual_seed(0)
                    self.q_queue = Queue()
                    self.q_queue.put(
                        torch.randn(
                            [args.micro_batch_size, args.hidden_size, args.pipeline_qr_r],
                            generator=self.generator,
                            device=torch.cuda.current_device(),
                            dtype=torch.float64
                        )
                    )
                elif self.pipeline_compress_method == 'ef_power':
                    self.generator = torch.Generator(device=torch.cuda.current_device()).manual_seed(0)
                    self.q_queue = Queue()
                    self.q_queue.put(
                        torch.randn(
                            [args.micro_batch_size, args.hidden_size, args.pipeline_qr_r],
                            generator=self.generator,
                            device=torch.cuda.current_device(),
                            dtype=torch.float64
                        )
                    )
                    self.error_feedback = torch.zeros(
                        args.seq_length, args.micro_batch_size, args.hidden_size,
                        device=torch.cuda.current_device(),
                        dtype=torch.float64
                    )
                elif self.pipeline_compress_method == 'topk':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   device=torch.cuda.current_device(),
                                                   dtype=torch.int64)
                elif self.pipeline_compress_method == 'randk':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   device=torch.cuda.current_device(),
                                                   dtype=torch.int64)
                elif self.pipeline_compress_method == 'srht':
                    self.H_tensor = torch.tensor(hadamard(args.hidden_size), dtype=torch.float16).cuda()
                elif self.pipeline_compress_method == 'topk_feedback':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   device=torch.cuda.current_device(),
                                                   dtype=torch.int64)
                    self.error_feedback = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                      device=torch.cuda.current_device(),
                                                      dtype=args.params_dtype)
                elif self.pipeline_compress_method == 'randk_feedback':
                    self.bool_matrix = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                   device=torch.cuda.current_device(),
                                                   dtype=torch.int64)
                    self.error_feedback = torch.zeros(args.seq_length, args.micro_batch_size, args.hidden_size,
                                                      device=torch.cuda.current_device(),
                                                      dtype=args.params_dtype)

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        args = get_args()
        if self.is_pipeline_compress and not args.skip_compression:
            if self.pipeline_compress_method == 'ae':
                output = F.linear(output, self.encoder)
            elif self.pipeline_compress_method == "topk_int":
                value, indices, _, _ = topk.encoder(output, k=self.pipeline_k)
                output = [value, indices]
            elif self.pipeline_compress_method == "randk_int":
                value, indices, _, _ = randk.encoder(output, k=self.pipeline_k)
                output = [value, indices]
            elif self.pipeline_compress_method == "power":
                hidden_states = hidden_states.permute([1, 0, 2])
                q_prev = self.q_queue.get()
                hidden_states = hidden_states.to(dtype=torch.float64, memory_format=torch.contiguous_format)
                p = torch.bmm(hidden_states.permute([1, 0, 2]), q_prev[:args.micro_batch_size, :, :])
                p_hat = torch.linalg.qr(p).Q
                q_next = torch.bmm(hidden_states.permute([1, 2, 0]), p_hat).detach()
                if q_next.size()[0] == q_prev.size()[0]:
                    self.q_queue.put(q_next)
                else:
                    self.q_queue.put(q_prev)
                p_hat = p_hat.to(dtype=args.params_dtype, memory_format=torch.contiguous_format)
                q_next = q_next.to(dtype=args.params_dtype, memory_format=torch.contiguous_format)
                output = [p_hat, q_next]
            elif self.pipeline_compress_method == "ef_power":
                hidden_states = hidden_states.permute([1, 0, 2])
                q_prev = self.q_queue.get()
                hidden_states = hidden_states.to(dtype=torch.float64, memory_format=torch.contiguous_format)
                hidden_states_last = hidden_states.clone()
                if hidden_states.size()[1] == self.error_feedback.size()[1]:
                    hidden_states = hidden_states + self.error_feedback
                p = torch.bmm(hidden_states.permute([1, 0, 2]), q_prev[:args.micro_batch_size, :, :])
                p_hat = torch.linalg.qr(p).Q
                q_next = torch.bmm(hidden_states.permute([1, 2, 0]), p_hat).detach()
                if q_next.size()[0] == q_prev.size()[0]:
                    self.q_queue.put(q_next)
                else:
                    self.q_queue.put(q_prev)
                decompose_matrix = torch.bmm(p_hat, q_next.permute([0, 2, 1]))
                decompose_matrix = decompose_matrix.permute([1, 0, 2])
                if decompose_matrix.size()[1] == hidden_states_last.size()[1]:
                    self.error_feedback = hidden_states_last.detach() - decompose_matrix.detach()
                p_hat = p_hat.to(dtype=args.params_dtype, memory_format=torch.contiguous_format)
                q_next = q_next.to(dtype=args.params_dtype, memory_format=torch.contiguous_format)
                output = [p_hat, q_next]
            elif self.pipeline_compress_method == 'topk':
                batch_size = output.size()[1]
                value, indices, input_abs_size, input_abs_seq_size = topk.encoder(output, k=self.pipeline_k)
                self.bool_matrix.zero_()
                bool_matrix = self.bool_matrix[:, :batch_size, :].detach()
                bool_matrix = bool_matrix.reshape(-1, )
                bool_matrix[indices] = 1
                bool_matrix = bool_matrix.reshape(input_abs_size)
                loc = torch.nonzero(bool_matrix)
                value = value.reshape(-1, 1)
                output = torch.cat((loc, value), dim=1)
            elif self.pipeline_compress_method == 'randk':
                batch_size = output.size()[1]
                value, indices, input_abs_size, input_abs_seq_size = randk.encoder(output, k=self.pipeline_k)
                self.bool_matrix.zero_()
                bool_matrix = self.bool_matrix[:, :batch_size, :].detach()
                bool_matrix = bool_matrix.reshape(-1, )
                bool_matrix[indices] = 1
                bool_matrix = bool_matrix.reshape(input_abs_size)
                loc = torch.nonzero(bool_matrix)
                value = value.reshape(-1, 1)
                output = torch.cat((loc, value), dim=1)
            elif self.pipeline_compress_method == 'topk_old':
                value, indices, _, _ = topk.encoder(output, k=self.pipeline_k)
                value = value.to(torch.float32)
                indices = indices.to(torch.float32)
                output = torch.stack((value, indices), 0)
            elif self.pipeline_compress_method == 'randk_old':
                value, indices, _, _ = topk.encoder(output, k=self.pipeline_k)
                value = value.to(torch.float32)
                indices = indices.to(torch.float32)
                output = torch.stack((value, indices), 0)
            elif self.pipeline_compress_method == 'topk_feedback':
                batch_size = output.size()[1]
                output.data = output.data + self.error_feedback[:, :batch_size, :].data
                value, indices, input_abs_size, input_abs_seq_size = topk.encoder(output, k=self.pipeline_k)
                topk_sparse = torch.sparse_coo_tensor(indices.unsqueeze(0), value, input_abs_seq_size)
                topk_dense = topk_sparse.to_dense()
                topk_res = torch.reshape(topk_dense, input_abs_size)
                self.error_feedback[:, :batch_size, :].data = output.data - topk_res.data
                self.bool_matrix.zero_()
                bool_matrix = self.bool_matrix[:, :batch_size, :].detach()
                bool_matrix = bool_matrix.reshape(-1, )
                bool_matrix[indices] = 1
                bool_matrix = bool_matrix.reshape(input_abs_size)
                loc = torch.nonzero(bool_matrix)
                value = value.reshape(-1, 1)
                output = torch.cat((loc, value), dim=1)
            elif self.pipeline_compress_method == 'randk_feedback':
                batch_size = output.size()[1]
                output.data = output.data + self.error_feedback[:, :batch_size, :].data
                value, indices, input_abs_size, input_abs_seq_size = randk.encoder(output, k=self.pipeline_k)
                topk_sparse = torch.sparse_coo_tensor(indices.unsqueeze(0), value, input_abs_seq_size)
                topk_dense = topk_sparse.to_dense()
                topk_res = torch.reshape(topk_dense, input_abs_size)
                self.error_feedback[:, :batch_size, :].data = output.data - topk_res.data
                self.bool_matrix.zero_()
                bool_matrix = self.bool_matrix[:, :batch_size, :].detach()
                bool_matrix = bool_matrix.reshape(-1, )
                bool_matrix[indices] = 1
                bool_matrix = bool_matrix.reshape(input_abs_size)
                loc = torch.nonzero(bool_matrix)
                value = value.reshape(-1, 1)
                output = torch.cat((loc, value), dim=1)
            elif self.pipeline_compress_method == 'srht':
                compress_output, S = srht.encoder(output, self.H_tensor,
                                                  d=args.hidden_size, m=args.pipeline_m)
                compress_output = torch.reshape(
                    compress_output, (args.seq_length * args.micro_batch_size, args.pipeline_m)
                )
                output = torch.cat((compress_output, S.T), 0)
            elif self.pipeline_compress_method == 'ct':
                compress_output, S = ct.encoder(output,
                                                d=args.hidden_size, m=args.pipeline_m)
                compress_output = torch.reshape(
                    compress_output, (args.seq_length * args.micro_batch_size, args.pipeline_m)
                )
                output = torch.cat((compress_output, S.T), 0)
            elif self.pipeline_compress_method == 'qr':
                Q = torch.randn(args.hidden_size,
                                self.pipeline_qr_r, dtype=torch.float16).cuda()
                P = torch.matmul(output, Q)
                P = P.to(torch.float32)
                P = P.permute(1, 0, 2)
                P_hat = torch.linalg.qr(P).Q
                P_hat = P_hat.to(torch.float16)
                Q = torch.matmul(output.permute(1, 2, 0), P_hat)
                output = torch.cat((P_hat, Q), 1)
            elif self.pipeline_compress_method == "quantize":
                if args.pipeline_bits == 8:
                    output.data, scale = quantize.compress_8bit(output.detach())
                elif args.pipeline_bits == 4:
                    output.data, scale = quantize.compress_4bit(output.detach())
                elif args.pipeline_bits == 2:
                    output.data, scale = quantize.compress_2bit(output.detach())
                else:
                    raise ValueError("pipeline bits is not correct")
                output = [output, scale]
            elif self.pipeline_compress_method == 'quantize_float':
                if args.pipeline_bits == 8:
                    compress_set = quantize.compress_8bit(output)
                    value = compress_set[0].to(torch.float16)
                    scaler = compress_set[1].to(torch.float16)
                    scaler = torch.flatten(scaler)
                    value_size = value.size()
                    value = value.reshape(value_size[0] * value_size[1], value_size[2])
                    scaler = scaler.reshape(1, value_size[2])
                    output = torch.cat((value, scaler), dim=0)
                elif args.pipeline_bits == 4:
                    compress_set = quantize.compress_4bit(output)
                    value = compress_set[0].to(torch.float16)
                    scaler = compress_set[1].to(torch.float16)
                    scaler = torch.flatten(scaler)
                    value_size = value.size()
                    value = value.reshape(value_size[0] * value_size[1], value_size[2])
                    scaler = scaler.reshape(2, value_size[2])
                    output = torch.cat((value, scaler), dim=0)
                elif args.pipeline_bits == 2:
                    compress_set = quantize.compress_2bit(output)
                    value = compress_set[0].to(torch.float16)
                    scaler = compress_set[1].to(torch.float16)
                    scaler = torch.flatten(scaler)
                    value_size = value.size()
                    value = value.reshape(value_size[0] * value_size[1], value_size[2])
                    scaler = scaler.reshape(4, value_size[2])
                    output = torch.cat((value, scaler), dim=0)
                else:
                    raise ValueError("tensor bits is error")
            elif self.pipeline_compress_method == 'quantize_old':
                output = output.to(torch.int16)
                output = output.to(torch.int8)
            else:
                raise ValueError("Pipeline Compression Method is Wrong")
            # output = F.normalize(output)

        return output


class ParallelTransformerDecoderLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(ParallelTransformerDecoderLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.is_pipeline_compress = args.is_pipeline_compress
        self.pipeline_compress_method = args.pipeline_compress_method
        self.pipeline_ae_dim = args.pipeline_ae_dim
        self.pipeline_qr_r = args.pipeline_qr_r
        self.pipeline_k = args.pipeline_k

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(init_method, output_layer_init_method, layer_number)
        else:
            self.mlp = ParallelMLP(init_method, output_layer_init_method, layer_number)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
            nullcontext if use_nvfuser else torch.enable_grad

        if args.use_cpu_initialization:
            if self.is_pipeline_compress:
                if self.pipeline_compress_method == 'ae':
                    # torch.nn.init.xavier_uniform_ is used to avoid exploding gradient problem
                    self.decoder = Parameter(torch.nn.init.xavier_uniform_(
                        torch.empty(args.hidden_size, args.pipeline_ae_dim,
                                    dtype=args.params_dtype))
                    )
        else:
            if self.is_pipeline_compress:
                if self.pipeline_compress_method == 'ae':
                    # torch.nn.init.xavier_uniform_ is used to avoid exploding gradient problem
                    self.decoder = Parameter(torch.nn.init.xavier_uniform_(
                        torch.empty(args.hidden_size, args.pipeline_ae_dim,
                                    device=torch.cuda.current_device(),
                                    dtype=args.params_dtype))
                    )

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]
        args = get_args()
        if self.is_pipeline_compress and not args.skip_compression:
            if self.pipeline_compress_method == 'ae':
                hidden_states = F.linear(hidden_states, self.decoder)
            elif self.pipeline_compress_method == "topk_int":
                value, indices = hidden_states[0], hidden_states[1]
                input_abs_size = torch.Size([args.micro_batch_size,
                                             int(args.image_size / args.patch_size) ** 2 + 1,
                                             args.hidden_size])
                input_abs_seq_size = torch.Size([args.micro_batch_size *
                                                 (int(args.image_size / args.patch_size) ** 2 + 1) *
                                                 args.hidden_size])
                hidden_states = topk.decoder(value, indices, input_abs_size, input_abs_seq_size)
            elif self.pipeline_compress_method == "randk_int":
                value, indices = hidden_states[0], hidden_states[1]
                input_abs_size = torch.Size([args.micro_batch_size,
                                             int(args.image_size / args.patch_size) ** 2 + 1,
                                             args.hidden_size])
                input_abs_seq_size = torch.Size([args.micro_batch_size *
                                                 (int(args.image_size / args.patch_size) ** 2 + 1) *
                                                 args.hidden_size])
                hidden_states = randk.decoder(value, indices, input_abs_size, input_abs_seq_size)
            elif self.pipeline_compress_method == "power":
                p_hat, q = hidden_states[0], hidden_states[1]
                p_hat = p_hat.to(dtype=torch.float64, memory_format=torch.contiguous_format)
                q = q.to(dtype=torch.float64, memory_format=torch.contiguous_format)
                hidden_states = torch.bmm(p_hat, q.permute([0, 2, 1]))
                hidden_states = hidden_states.to(dtype=args.params_dtype, memory_format=torch.contiguous_format)
            elif self.pipeline_compress_method == "ef_power":
                p_hat, q = hidden_states[0], hidden_states[1]
                p_hat = p_hat.to(dtype=torch.float64, memory_format=torch.contiguous_format)
                q = q.to(dtype=torch.float64, memory_format=torch.contiguous_format)
                hidden_states = torch.bmm(p_hat, q.permute([0, 2, 1]))
                hidden_states = hidden_states.to(dtype=args.params_dtype, memory_format=torch.contiguous_format)
            elif self.pipeline_compress_method == 'topk':
                input_abs_size = torch.Size([args.seq_length, args.micro_batch_size, args.hidden_size])
                loc = hidden_states[:, :3]
                value = hidden_states[:, 3]
                hidden_states = torch.sparse_coo_tensor(loc.T, value, input_abs_size).to_dense()
            elif self.pipeline_compress_method == 'randk':
                input_abs_size = torch.Size([args.seq_length, args.micro_batch_size, args.hidden_size])
                loc = hidden_states[:, :3]
                value = hidden_states[:, 3]
                hidden_states = torch.sparse_coo_tensor(loc.T, value, input_abs_size).to_dense()
            elif self.pipeline_compress_method == "topk_old":
                value, indices = hidden_states[0].to(torch.float16), hidden_states[1].to(torch.int64)
                input_abs_size = torch.Size([args.seq_length, args.micro_batch_size, args.hidden_size])
                input_abs_seq_size = torch.Size([args.seq_length *
                                                 args.micro_batch_size *
                                                 args.hidden_size])
                hidden_states = topk.decoder(value, indices, input_abs_size, input_abs_seq_size)
            elif self.pipeline_compress_method == "randk_old":
                value, indices = hidden_states[0].to(torch.float16), hidden_states[1].to(torch.int64)
                input_abs_size = torch.Size([args.seq_length, args.micro_batch_size, args.hidden_size])
                input_abs_seq_size = torch.Size([args.seq_length *
                                                 args.micro_batch_size *
                                                 args.hidden_size])
                hidden_states = randk.decoder(value, indices, input_abs_size, input_abs_seq_size)
            elif self.pipeline_compress_method == 'topk_feedback':
                input_abs_size = torch.Size([args.seq_length, args.micro_batch_size, args.hidden_size])
                loc = hidden_states[:, :3]
                value = hidden_states[:, 3]
                hidden_states = torch.sparse_coo_tensor(loc.T, value, input_abs_size).to_dense()
            elif self.pipeline_compress_method == 'randk_feedback':
                input_abs_size = torch.Size([args.seq_length, args.micro_batch_size, args.hidden_size])
                loc = hidden_states[:, :3]
                value = hidden_states[:, 3]
                hidden_states = torch.sparse_coo_tensor(loc.T, value, input_abs_size).to_dense()
            elif self.pipeline_compress_method == 'srht':
                output_compress, S_T = hidden_states[:args.seq_length * args.micro_batch_size, :], \
                                       hidden_states[args.seq_length * args.micro_batch_size:, :]
                output_compress = torch.reshape(
                    output_compress, (args.seq_length, args.micro_batch_size, args.pipeline_m)
                )
                hidden_states = srht.decoder(output_compress, S_T.T)
            elif self.pipeline_compress_method == 'ct':
                output_compress, S_T = hidden_states[:args.seq_length * args.micro_batch_size, :], \
                                       hidden_states[args.seq_length * args.micro_batch_size:, :]
                output_compress = torch.reshape(
                    output_compress, (args.seq_length, args.micro_batch_size, args.pipeline_m)
                )
                hidden_states = ct.decoder(output_compress, S_T.T)
            elif self.pipeline_compress_method == 'qr':
                split = torch.split(hidden_states, [args.seq_length, args.hidden_size], 1)
                P_hat = split[0]
                Q = split[1]
                hidden_states = torch.matmul(P_hat, Q.permute(0, 2, 1))
                hidden_states = hidden_states.permute(1, 0, 2)
            elif self.pipeline_compress_method == "quantize":
                if args.pipeline_bits == 8:
                    hidden_states[0].data = quantize.decompress_8bit(hidden_states[0].detach(), hidden_states[1])
                elif args.pipeline_bits == 4:
                    hidden_states[0].data = quantize.decompress_4bit(hidden_states[0].detach(), hidden_states[1])
                elif args.pipeline_bits == 2:
                    hidden_states[0].data = quantize.decompress_2bit(hidden_states[0].detach(), hidden_states[1])
                else:
                    raise ValueError("pipeline bits is not correct")
                hidden_states = hidden_states[0]
            elif self.pipeline_compress_method == 'quantize_float':
                if args.pipeline_bits == 8:
                    value = hidden_states[:args.seq_length * args.micro_batch_size, :]
                    scale = hidden_states[args.seq_length * args.micro_batch_size:, :]
                    value = value.reshape(args.seq_length, args.micro_batch_size, args.hidden_size)
                    scale = scale.reshape(1, -1).unsqueeze(0)
                    hidden_states = quantize.decompress_8bit(value, scale)
                elif args.pipeline_bits == 4:
                    value = hidden_states[:args.seq_length * args.micro_batch_size, :]
                    scale = hidden_states[args.seq_length * args.micro_batch_size:, :]
                    value = value.reshape(args.seq_length, args.micro_batch_size, int(args.hidden_size / 2))
                    scale = scale.reshape(1, -1).unsqueeze(0)
                    hidden_states = quantize.decompress_4bit(value, scale)
                elif args.pipeline_bits == 2:
                    value = hidden_states[:args.seq_length * args.micro_batch_size, :]
                    scale = hidden_states[args.seq_length * args.micro_batch_size:, :]
                    value = value.reshape(args.seq_length, args.micro_batch_size, int(args.hidden_size / 4))
                    scale = scale.reshape(1, -1).unsqueeze(0)
                    hidden_states = quantize.decompress_2bit(value, scale)
                else:
                    raise ValueError("tensor bits is error")
            elif self.pipeline_compress_method == 'quantize_old':
                hidden_states = hidden_states.to(torch.int16)
                hidden_states = hidden_states.to(torch.float16)
            else:
                raise ValueError("Pipeline Compression Method is Wrong")
            # hidden_states = F.normalize(hidden_states)

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        return output


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True, post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = args.model_type
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.is_vision_train = args.is_vision_train

        # Store activation checkpoiting flag.
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = \
            args.distribute_saved_activations and not args.sequence_parallel

        self.sequence_parallel = args.sequence_parallel

        # Number of layers.
        self.num_layers = mpu.get_num_layers(
            args, args.model_type == ModelType.encoder_and_decoder)

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, args.num_layers)]

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])

        def build_encoder_layer(layer_number):
            return ParallelTransformerEncoderLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])

        def build_decoder_layer(layer_number):
            return ParallelTransformerDecoderLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                    args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                     (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        else:
            if args.is_pretrain_single_machine and args.is_pipeline_compress:
                layers_list = []
                for i in range(self.num_layers):
                    if i + 1 == 12 or i + 1 == 18:
                        layers_list.append(build_encoder_layer(i + 1 + offset))
                    elif i + 1 == 13 or i + 1 == 19:
                        layers_list.append(build_decoder_layer(i + 1 + offset))
                    else:
                        layers_list.append(build_layer(i + 1 + offset))
                    self.layers = torch.nn.ModuleList(layers_list)
            else:
                if args.is_pipeline_compress:
                    layers_list = []
                    # we can change the threshold for pipeline model parallel rank
                    # to custom the number of compression
                    start_rank = args.start_pipeline_compress_rank
                    if mpu.get_pipeline_model_parallel_rank() == start_rank:
                        for i in range(self.num_layers):
                            if i == self.num_layers - 1:
                                layers_list.append(build_encoder_layer(i + 1 + offset))
                            else:
                                layers_list.append(build_layer(i + 1 + offset))
                    elif mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1:
                        for i in range(self.num_layers):
                            if i == 0:
                                layers_list.append(build_decoder_layer(i + 1 + offset))
                            else:
                                layers_list.append(build_layer(i + 1 + offset))
                    elif mpu.get_pipeline_model_parallel_rank() < start_rank:
                        for i in range(self.num_layers):
                            layers_list.append(build_layer(i + 1 + offset))
                    else:
                        for i in range(self.num_layers):
                            if i == 0:
                                layers_list.append(build_decoder_layer(i + 1 + offset))
                            elif i == self.num_layers - 1:
                                layers_list.append(build_encoder_layer(i + 1 + offset))
                            else:
                                layers_list.append(build_layer(i + 1 + offset))
                    self.layers = torch.nn.ModuleList(layers_list)
                else:
                    self.layers = torch.nn.ModuleList(
                        [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                return x_

            return custom_forward

        if self.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                hidden_states = mpu.checkpoint(
                    custom(l, l + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                l += self.recompute_num_layers

        elif self.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    hidden_states = mpu.checkpoint(
                        custom(l, l + 1),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                else:
                    hidden_states = custom(l, l + 1)(
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        if self.is_vision_train:
            if isinstance(hidden_states, list):
                if len(hidden_states) == 1:
                    hidden_states = mpu.make_viewless_tensor(
                        hidden_states[0],
                        requires_grad=True,
                        keep_graph=True,
                    )
                elif len(hidden_states) == 2:
                    hidden_states[0], hidden_states[1] = mpu.make_viewless_tensor(
                        hidden_states[0],
                        requires_grad=True,
                        keep_graph=True,
                    ), mpu.make_viewless_tensor(
                        hidden_states[1],
                        requires_grad=False,
                        keep_graph=True,
                    )
            else:
                hidden_states = mpu.make_viewless_tensor(
                    hidden_states,
                    requires_grad=True,
                    keep_graph=True,
                )
        else:
            if isinstance(hidden_states, list):
                hidden_states[0], hidden_states[1] = mpu.make_viewless_tensor(
                    hidden_states[0],
                    requires_grad=True,
                    keep_graph=True,
                ), mpu.make_viewless_tensor(
                    hidden_states[1],
                    requires_grad=False,
                    keep_graph=True,
                )
            else:
                hidden_states = mpu.make_viewless_tensor(
                    hidden_states,
                    requires_grad=True,
                    keep_graph=True,
                )

        if self.sequence_parallel:
            rng_context = mpu.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # Forward pass.
            if self.recompute_granularity == 'full':
                hidden_states = self._checkpointed_forward(hidden_states,
                                                           attention_mask,
                                                           encoder_output,
                                                           enc_dec_attn_mask)
            else:
                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(
                        hidden_states,
                        attention_mask,
                        encoder_output=encoder_output,
                        enc_dec_attn_mask=enc_dec_attn_mask,
                        inference_params=inference_params)

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
