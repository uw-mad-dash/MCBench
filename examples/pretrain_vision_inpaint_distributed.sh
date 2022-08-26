#!/bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="../tiny-imagenet-200/train \
           ../tiny-imagenet-200/val"
CHECKPOINT_PATH=checkpoints/vision_inpaint

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../pretrain_vision_inpaint.py \
       --is-vision-train True \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 768 \
       --vision-backbone-type vit \
       --num-attention-heads 12 \
       --encoder-seq-length 512 \
       --decoder-seq-length 512 \
       --micro-batch-size 8 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --lr-decay-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --is-pipeline-compress False \
       --pipeline-compress-method randk \
       --pipeline-ae-dim 1024 \
       --pipeline-qr-r 10 \
       --pipeline-k 10000 \
       --pipeline-m 50 \
       --is-tensor-compress False \
       --tensor-compress-method ae \
       --tensor-ae-dim 100 \
       --tensor-qr-r 10 \
       --tensor-k 10 \
       --tensor-m 50 \
