#!/bin/bash

# compress method in [ae, quantize, topk, randk, topk_feedback, randk_feedback, qr]

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $1 \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATA_PATH=../my-bert-wiki-and-book_text_sentence
VOCAB_FILE=../bert-large-cased-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_pretrain_local

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../pretrain_bert.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 256 \
       --global-batch-size 1024 \
       --seq-length 128 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 969,30,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 0.00001 \
       --lr-decay-iters 1000000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.01 \
       --log-interval 10000 \
       --save-interval 50000 \
       --eval-interval 10000 \
       --eval-iters 100 \
       --fp16 \
       --is-pipeline-compress False \
       --pipeline-compress-method topk \
       --pipeline-ae-dim 1024 \
       --pipeline-qr-r 10 \
       --pipeline-k 10000 \
       --pipeline-m 50 \
       --pipeline-bits 8 \
       --start-pipeline-compress-rank 1 \
       --is-tensor-compress False \
       --tensor-compress-method topk \
       --tensor-ae-dim 50 \
       --tensor-qr-r 10 \
       --tensor-k 10000 \
       --tensor-m 50 \
       --tensor-bits 8 \
       --start-tensor-compress-layer 12 \
#               --no-masked-softmax-fusion # My environment (orca) can't handle this, feel free to comment it in AWS