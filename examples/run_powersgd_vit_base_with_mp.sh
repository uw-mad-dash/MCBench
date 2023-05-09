#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

PRETRAINED_CHECKPOINT=checkpoints/vision_classify_base_patch32/split_2t_2p
PATCH_SIZE=32
MICRO_BATCH_SIZE=512
LR=0.03

for method in 'power'
#'ef_power'
do
  for r in 50 100
  do
    for dataset in 'cifar10' 'cifar100'
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task classify \
               --dataset-name 'cifar10' \
               --num-classes 10 \
               --epochs 100 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 12 \
               --hidden-size 768 \
               --num-attention-heads 12 \
               --patch-size $PATCH_SIZE \
               --encoder-seq-length 512 \
               --decoder-seq-length 512 \
               --micro-batch-size $MICRO_BATCH_SIZE \
               --max-position-embeddings 512 \
               --optimizer 'sgd' \
               --lr $LR \
               --lr-decay-style 'cosine' \
               --weight-decay 0.0 \
               --save-interval 500000 \
               --log-interval 100 \
               --eval-interval 1000 \
               --eval-iters 100 \
               --layernorm-epsilon 1e-12 \
               --hidden-dropout 0.0 \
               --attention-dropout 0.0 \
               --fp16 \
               --is-pipeline-compress True \
               --pipeline-compress-method $method \
               --pipeline-qr-r $r \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-qr-r $r \
               --start-tensor-compress-layer 6
    done
  done
done


