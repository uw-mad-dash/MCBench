#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/CoLA/train.tsv"
VALID_DATA="../glue_data/CoLA/dev.tsv"
VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_4p
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/bert_345m_cola

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 4 \
               --task CoLA \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs 3 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --micro-batch-size 8 \
               --lr 5.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --save-interval 500000 \
               --save $CHECKPOINT_PATH \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --fp16 \
               --is-pipeline-compress True \
               --pipeline-compress-method topk_int \
               --pipeline-ae-dim 50 \
               --pipeline-qr-r 10 \
               --pipeline-k 400000 \
               --pipeline-m 50 \
               --pipeline-bits 2 \
               --start-pipeline-compress-rank 1 \
               --is-tensor-compress False \
               --tensor-compress-method ae \
               --tensor-ae-dim 50 \
               --tensor-qr-r 10 \
               --tensor-k 400000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 12 \

