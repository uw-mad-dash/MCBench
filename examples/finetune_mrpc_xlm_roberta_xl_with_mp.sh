#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/MRPC/msr_paraphrase_train.txt"
VALID_DATA="../glue_data/MRPC/msr_paraphrase_test.txt"
VOCAB_FILE="../xlm-roberta-xl-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/xlm-roberta-xl/split_4t
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/xlm-roberta-xl

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main_hf.py \
               --tensor-model-parallel-size 4 \
               --pipeline-model-parallel-size 1 \
               --task MRPC \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type XLM-Roberta-Tokenizer \
               --vocab-file $VOCAB_FILE \
               --epochs 5 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 36 \
               --hidden-size 2560 \
               --num-attention-heads 32 \
               --micro-batch-size 16 \
               --lr 5.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 512 \
               --max-position-embeddings 514 \
               --save-interval 500000 \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --fp16 \
               --is-pipeline-compress False \
               --pipeline-compress-method quantize \
               --pipeline-ae-dim 256 \
               --pipeline-qr-r 128 \
               --pipeline-k 1000000 \
               --pipeline-m 50 \
               --pipeline-bits 2 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method ae \
               --tensor-ae-dim 256 \
               --tensor-qr-r 256 \
               --tensor-k 1000000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 18 \

