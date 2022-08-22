#!/bin/bash

TARGET_PIPELINE_MODEL_PARALLEL_SIZE=1
TARGET_TENSOR_MODEL_PARALLEL_SIZE=4

VOCAB_FILE=../bert-large-cased-vocab.txt
CHECKPOINT_PATH=../examples/checkpoints/bert_345m

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


python -m torch.distributed.launch $DISTRIBUTED_ARGS split_mp_partitions.py \
        --model-type BERT \
        --pipeline-model-parallel-size 1 \
        --tensor-model-parallel-size 4 \
        --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
        --target-tensor-model-parallel-size $TARGET_TENSOR_MODEL_PARALLEL_SIZE \
        --tokenizer-type BertWordPieceLowerCase \
        --vocab-file $VOCAB_FILE \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH/split_4t \
        --task MNLI \
        --micro-batch-size 1 \
        --save-interval 500000