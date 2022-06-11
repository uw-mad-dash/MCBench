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

"""Split model parallel partitions."""

import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch

from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import get_checkpoint_version
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.global_vars import set_global_variables, get_args
from megatron.global_vars import rebuild_tokenizer

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.utils import unwrap_model

from torch.nn.parallel import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

from megatron import (get_args,
                      mpu,
                      print_rank_0,
                      utils)

import numpy as np
import random

def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    print("dirname: ", dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_rng_state():
    """ collect rng state across data parallel ranks """
    args = get_args()
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': mpu.get_cuda_rng_tracker().get_states()}

    rng_state_list = None
    if torch.distributed.is_initialized() and \
            mpu.get_data_parallel_world_size() > 1 and \
            args.data_parallel_random_init:
        rng_state_list = \
            [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]

    return rng_state_list


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = utils.unwrap_model(model)

    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    # collect rng state across data parallel ranks
    rng_state = get_rng_state()

    # Arguments, iteration, and model.
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = iteration
    if len(model) == 1:
        # state_dict['model'] = model[0].state_dict()
        state_dict['model'] = model[0].state_dict_for_save_checkpoint()
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict['optimizer'] = optimizer.state_dict()
        if opt_param_scheduler is not None:
            state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

    # RNG states.
    if not args.no_save_rng:
        state_dict["rng_state"] = rng_state

    # Save.
    checkpoint_name = get_checkpoint_name(args.save, iteration, True)
    ensure_directory_exists(checkpoint_name)
    torch.save(state_dict, checkpoint_name)

    print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    tracker_filename = get_checkpoint_tracker_filename(args.save)
    print("tracker_filename: ", tracker_filename)
    with open(tracker_filename, 'w') as f:
        f.write("release")


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')
    group.add_argument('--valid-data', nargs='*', default=None,
                       help='path(s) to the validation data.')
    group.add_argument('--overlapping-eval', type=int, default=32,
                       help='Sliding window for overlapping evaluation.')
    group.add_argument('--strict-lambada', action='store_true',
                       help='Use more difficult formulation of lambada.')
    # Retriever args
    group.add_argument('--qa-data-dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')
    group.add_argument('--qa-data-test', type=str, default=None,
                       help='Path to the QA dataset test file.')

    # Faiss arguments for retriever
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--faiss-match', type=str, default='string', \
                        choices=['regex', 'string'], help="Answer matching '\
                        'logic type")
    group.add_argument('--faiss-topk-retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')

    # finetune for retriever
    group.add_argument('--eval-micro-batch-size', type=int, default=None,
                       help='Eval Batch size per model instance (local batch '
                            'size). Global batch size is local batch size '
                            'times data parallel size.')
    group.add_argument('--train-with-neg', action='store_true',
                       help='Whether to use negative examples during model '
                        'training')
    group.add_argument('--train-hard-neg', type=int, default=0,
                       help='Number of hard negative exmaples to use during '
                        'training')


    # parameters for Av.rank validation method
    # Following options/arguments have been taken directly from DPR codebase
    group.add_argument('--val-av-rank-hard-neg', type=int, default=30,
                        help='Av.rank validation: how many hard negatives to'
                        ' take from each question pool')
    group.add_argument('--val-av-rank-other-neg', type=int, default=30,
                        help='Av.rank validation: how many other negatives to'
                        ' take from each question pool')

    group.add_argument('--model-type', type=str, required=True,
                       choices=['BERT', 'GPT', 'RACE', 'MNLI', 'QQP'],
                       help='Type of the mdoel.')
    group.add_argument('--target-pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism in output model.')
    group.add_argument('--target-tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism in output model.')


    return parser


def split_into_partitions(tensor, num_partitions, partition_dim, stride):

    per_partition_size = mpu.utils.divide(tensor.size(partition_dim),
                                          num_partitions)
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size, stride)

    partitions_list = torch.split(tensor,
                                  per_partition_per_stride_size,
                                  dim=partition_dim)

    partitions = []
    for i in range(num_partitions):
        partition = torch.cat(partitions_list[i::num_partitions],
                              dim=partition_dim)
        partitions.append(partition)

    return partitions


def get_model(model_type):
    if model_type == 'BERT':
        from pretrain_bert import model_provider
    elif model_type == 'GPT':
        from pretrain_gpt import model_provider
    elif model_type == 'RACE':
        from tasks.race.finetune import model_provider
    elif model_type == ['MNLI', 'QQP']:
        num_classes = 2
        if model_type == 'MNLI':
            num_classes = 3
        from megatron.model.classification import Classification
        def model_provider():
            return Classification(num_classes=num_classes, num_tokentypes=2)
    else:
        raise Exception('unrecognized model type: {}'.format(model_type))

    model = model_provider()
    model = model.half()

    return model


def get_parallel_checkpoint_name(path):
    print("path: ", path)
    tracker_filename = get_checkpoint_tracker_filename(path)
    print("tracker_filename: ", tracker_filename)
    iteration = 0
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
    # assert iteration > 0
    checkpoint_name = get_checkpoint_name(path, iteration)

    return checkpoint_name, iteration


def get_mp_split_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='mp split')

    group.add_argument('--model-type', type=str, required=True,
                       choices=['BERT', 'GPT', 'RACE', 'MNLI', 'QQP'],
                       help='Type of the mdoel.')
    group.add_argument('--target-pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism in output model.')
    group.add_argument('--target-tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism in output model.')

    return parser


def main():
    initialize_megatron(extra_args_provider=get_tasks_args)
    print("finish initialize")
    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    os.environ["WORLD_SIZE"] = f'{2**31}'

    # Args
    # set_global_variables(extra_args_provider=get_mp_split_args,
    #                      args_defaults={'use_cpu_initialization': True,
    #                                     'micro_batch_size': 1,
    #                                     'no_load_optim': True,
    #                                     'no_load_rng': True,
    #                                     'no_save_optim': True,
    #                                     'no_save_rng': True,
    #                                     'save_interval': 1})

    args = get_args()
    # fused_kernels.load(args)
    model_type = args.model_type
    tokenizer = rebuild_tokenizer(args)

    print('\n merging model parallel partitions ...')
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of tokens ................ {} '.format(tokenizer.vocab_size))
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden size ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(args.num_attention_heads))
    print('    maximum position embeddings ..... {}'.format(args.max_position_embeddings))

    org_tensor_model_parallel_rank = mpu.get_tensor_model_parallel_rank()
    org_pipeline_model_parallel_rank = mpu.get_pipeline_model_parallel_rank()
    # Full model.
    print('> building the full model ...')
    mpu.initialize.set_tensor_model_parallel_world_size(1)
    mpu.initialize.set_tensor_model_parallel_rank(0)
    mpu.initialize.set_pipeline_model_parallel_world_size(1)
    mpu.initialize.set_pipeline_model_parallel_rank(0)

    # load merged_model
    iteration = 0
    # double check here: where get_parallel_checkpoint_name could load checkpoint correctly
    checkpoint_name, iteration = get_parallel_checkpoint_name(args.load)
    merged_model = get_model(model_type)
    # for param_tensor in merged_model.state_dict():
    #     print(param_tensor, "\t", merged_model.state_dict()[param_tensor].size())
    merged_model = [merged_model]
    # unwrapped_model = unwrap_model(merged_model,
    #                                (torchDDP, LocalDDP, Float16Module))
    print(f'> loading {checkpoint_name} ...')
    load_checkpoint(merged_model, None, None)
    print(f'> checkpoint version {get_checkpoint_version()}')

    partitions = []
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size

    assert args.num_layers % args.pipeline_model_parallel_size == 0, \
        'num_layers must be divisible by target pipeline model parallel size'
    layers_per_part = args.num_layers // args.pipeline_model_parallel_size

    tokenizer = rebuild_tokenizer(args)
    mpu.initialize.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    mpu.initialize.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)

    layer_re = re.compile('layers\.([0-9]+)')

    if args.pipeline_model_parallel_size > 1:
        merged_params = {}
        for name, merged_param in merged_model[0].named_parameters():
            merged_params[name] = merged_param

        for rank in range(args.pipeline_model_parallel_size):
            mpu.initialize.set_pipeline_model_parallel_rank(rank)
            if args.tensor_model_parallel_size > 1:
                for _ in range(args.tensor_model_parallel_size):
                    partitions.append([])
                for tensor_rank in range(args.tensor_model_parallel_size):
                    mpu.initialize.set_tensor_model_parallel_rank(tensor_rank)
                    model = get_model(model_type)
                    def update_layer_num(m):
                        # TODO! This assumes no interleaved pipeline execution
                        layer = int(m.group(1))
                        layer += rank * layers_per_part
                        return f'layers.{layer}'

                    for dst_name, partition_param in model.named_parameters():
                        # print("dst_name: ", dst_name)
                        # print("partition_param: ", partition_param.data.size())
                        if dst_name == "word_embeddings.weight":
                            # See comment in MegatronModule.initialize_word_embeddings()
                            src_name = "language_model.embedding.word_embeddings.weight"
                        else:
                            # Translate destination layer number (0-N for each partition)
                            # to source layer number (single-model layer number)
                            src_name = re.sub(layer_re, update_layer_num, dst_name)
                        # print(f" > copying {src_name} to {dst_name} in rank {rank}'s model")

                        # For the non-parallel parameters, simply copy the rank 0 values.
                        if not hasattr(merged_params[src_name], 'tensor_model_parallel'):
                            # print('     none-parallel parameter, simple copy from rank 0')
                            with torch.no_grad():
                                partition_param.data.copy_(merged_params[src_name].data)
                        # For parallel parameters, merge the values
                        else:
                            dim = merged_params[src_name].partition_dim
                            stride = merged_params[src_name].partition_stride
                            # print(f'     parallel parameter split with stride {stride} along '
                            #       f'dimention {dim}')
                            split_results = split_into_partitions(merged_params[src_name].data,
                                                                  args.tensor_model_parallel_size,
                                                                  dim,
                                                                  stride)

                            partition_param.data.copy_(split_results[tensor_rank])
                    partitions[tensor_rank].append(model)
            else:
                model = get_model(model_type)
                def update_layer_num(m):
                    # TODO! This assumes no interleaved pipeline execution
                    layer = int(m.group(1))
                    layer += rank * layers_per_part
                    return f'layers.{layer}'

                for dst_name, partition_param in model.named_parameters():
                    if dst_name == "word_embeddings.weight":
                        # See comment in MegatronModule.initialize_word_embeddings()
                        src_name = "language_model.embedding.word_embeddings.weight"
                    else:
                        # Translate destination layer number (0-N for each partition)
                        # to source layer number (single-model layer number)
                        src_name = re.sub(layer_re, update_layer_num, dst_name)
                    print(f" > copying {src_name} to {dst_name} in rank {rank}'s model")
                    partition_param.data.copy_(merged_params[src_name].data)

                partitions.append(model)
    else:
        partitions = [merged_model]

    if org_pipeline_model_parallel_rank == 0 and org_tensor_model_parallel_rank == 0:
        for tensor_rank in range(len(partitions)):
            for pipeline_rank in range(len(partitions[tensor_rank])):
                print(tensor_rank, pipeline_rank)
                mpu.initialize.set_tensor_model_parallel_rank(tensor_rank)
                mpu.initialize.set_pipeline_model_parallel_rank(pipeline_rank)
                model = partitions[tensor_rank][pipeline_rank]
                # for param_tensor in model.state_dict():
                #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                model = [model]
                print(f"> saving tensor_rank {tensor_rank} and pipeline_rank {pipeline_rank}'s model")
                save_checkpoint(iteration, model, None, None)


    # for rank, model in enumerate(partitions):
    #     mpu.initialize.set_pipeline_model_parallel_rank(rank)
    #     print(f"> saving rank {rank}'s model")
    #     save_checkpoint(iteration, model, None, None)

    print('done :-)')


if __name__ == '__main__':
    main()
