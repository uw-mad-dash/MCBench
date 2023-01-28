#!/bin/bash

TARGET_PIPELINE_MODEL_PARALLEL_SIZE=4
TARGET_TENSOR_MODEL_PARALLEL_SIZE=1

VOCAB_FILE=../bert-large-cased-vocab.txt
CHECKPOINT_PATH=../examples/checkpoints/bert_base_hf

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


python3 -m torch.distributed.launch $DISTRIBUTED_ARGS split_mp_partitions_single_hf.py \
        --model-type BERT \
        --pipeline-model-parallel-size 4 \
        --tensor-model-parallel-size 1 \
        --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
        --target-tensor-model-parallel-size $TARGET_TENSOR_MODEL_PARALLEL_SIZE \
        --tokenizer-type BertBaseHF \
        --vocab-file $VOCAB_FILE \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --seq-length 128 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH/split_4p \
        --task MNLI \
        --micro-batch-size 1 \
        --save-interval 500000