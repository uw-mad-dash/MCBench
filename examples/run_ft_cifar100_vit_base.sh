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

# when you change the number of patch, you need to modify the k value for topk_int
# for method in 'ae' 'quantize' 'topk_int' 'without'
for method in 'topk_int'
do
  if [ $method == 'ae' ];
  then
    for dim in 50 100
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task classify \
               --dataset-name 'cifar100' \
               --num-classes 100 \
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
               --pipeline-ae-dim $dim \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-ae-dim $dim \
               --start-tensor-compress-layer 6
    done
  elif [ $method == 'quantize' ];
  then
    for bits in 2 4 8
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task classify \
               --dataset-name 'cifar100' \
               --num-classes 100 \
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
               --pipeline-bits $bits \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-bits $bits \
               --start-tensor-compress-layer 6
    done
  elif [ $method == 'topk_int' ];
  then
    for k in 2560000 1280000 256000 512000
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task classify \
               --dataset-name 'cifar100' \
               --num-classes 100 \
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
               --pipeline-k $k \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-k $k \
               --start-tensor-compress-layer 6
    done
  elif [ $method == 'without' ];
  then
    for lr in 0.03
#    0.01 0.03 0.05 0.1 0.3 0.5
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task classify \
               --dataset-name 'cifar100' \
               --num-classes 100 \
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
               --lr $lr \
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
               --is-pipeline-compress False \
               --is-tensor-compress False
    done
  fi
done
