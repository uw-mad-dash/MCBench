#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6001"


#VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/vision_classify_large_patch16/split_2t_2p
#PRETRAINED_CHECKPOINT=checkpoints/vision_classify_base_patch16
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m
#CHECKPOINT_PATH=checkpoints/bert_base_hf_cola

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task classify \
               --dataset-name 'cifar10' \
               --num-classes 10 \
               --epochs 100 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --patch-size 16 \
               --encoder-seq-length 512 \
               --decoder-seq-length 512 \
               --micro-batch-size 512 \
               --max-position-embeddings 512 \
               --optimizer 'sgd' \
               --lr 0.03 \
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
               --pipeline-compress-method ae \
               --pipeline-ae-dim 100 \
               --pipeline-qr-r 10 \
               --pipeline-k 100 \
               --pipeline-m 50 \
               --pipeline-bits 8 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress False \
               --tensor-compress-method ae \
               --tensor-ae-dim 100 \
               --tensor-qr-r 10 \
               --tensor-k 80000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 12 \
#               --lr-warmup-fraction 0.065 \
#               --weight-decay 1.0e-1 \
