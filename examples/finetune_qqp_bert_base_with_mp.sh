#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/QQP/train.tsv"
VALID_DATA="../glue_data/QQP/dev.tsv"
VOCAB_FILE="../bert-base-cased-vocab.txt"
#VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_base_hf/split_2t_2p
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m
#CHECKPOINT_PATH=checkpoints/bert_345m_cola

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main_hf.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task QQP \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertBaseHF \
               --vocab-file $VOCAB_FILE \
               --epochs 3 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 12 \
               --hidden-size 768 \
               --num-attention-heads 12 \
               --micro-batch-size 32 \
               --lr 2.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 128 \
               --max-position-embeddings 512 \
               --save-interval 500000 \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --layernorm-epsilon 1e-12 \
               --fp16 \
               --is-pipeline-compress True \
               --pipeline-compress-method topk_int \
               --pipeline-ae-dim 50 \
               --pipeline-qr-r 10 \
               --pipeline-k 80000 \
               --pipeline-m 50 \
               --pipeline-bits 8 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method topk_int \
               --tensor-ae-dim 50 \
               --tensor-qr-r 10 \
               --tensor-k 80000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 6 \

