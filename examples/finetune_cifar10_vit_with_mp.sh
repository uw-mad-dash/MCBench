#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


#VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/vision_classify
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m
#CHECKPOINT_PATH=checkpoints/bert_base_hf_cola

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 1 \
               --task classify \
               --dataset-name "cifar10" \
               --num-classes 10 \
               --epochs 5 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 12 \
               --hidden-size 768 \
               --num-attention-heads 12 \
               --encoder-seq-length 512 \
               --decoder-seq-length 512 \
               --micro-batch-size 8 \
               --max-position-embeddings 512 \
               --lr 2.0e-5 \
               --lr-warmup-fraction 0.065 \
               --save-interval 500000 \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --layernorm-epsilon 1e-12 \
               --hidden-dropout 0.0 \
               --attention-dropout 0.0 \
               --is-pipeline-compress False \
               --pipeline-compress-method topk_int \
               --pipeline-ae-dim 50 \
               --pipeline-qr-r 10 \
               --pipeline-k 80000 \
               --pipeline-m 50 \
               --pipeline-bits 8 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress False \
               --tensor-compress-method topk_int \
               --tensor-ae-dim 50 \
               --tensor-qr-r 10 \
               --tensor-k 80000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 6 \

