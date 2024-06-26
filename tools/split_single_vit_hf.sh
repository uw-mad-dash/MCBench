#!/bin/bash

TARGET_PIPELINE_MODEL_PARALLEL_SIZE=2
TARGET_TENSOR_MODEL_PARALLEL_SIZE=2

CHECKPOINT_PATH=../examples/checkpoints/vision_classify_base_patch32

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


python3 -m torch.distributed.launch $DISTRIBUTED_ARGS split_mp_partitions_single_vit_hf.py \
        --model-type 'vit' \
        --vision-backbone-type 'vit' \
        --task classify \
        --finetune \
        --num-classes 10 \
        --dataset-name "cifar10" \
        --pipeline-model-parallel-size 2 \
        --tensor-model-parallel-size 2 \
        --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
        --target-tensor-model-parallel-size $TARGET_TENSOR_MODEL_PARALLEL_SIZE \
        --encoder-seq-length 512 \
        --decoder-seq-length 512 \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --patch-size 32 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH/split_2t_2p \
        --micro-batch-size 1 \
        --save-interval 500000