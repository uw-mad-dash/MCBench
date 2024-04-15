#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/MNLI/train.tsv"
VALID_DATA="../glue_data/MNLI/dev_matched.tsv \
            ../glue_data/MNLI/dev_mismatched.tsv"
VOCAB_FILE="../opt-3b-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/opt_3b/split_2t_2p
CHECKPOINT_PATH=checkpoints/opt_3b_mnli

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task MNLI \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type OPTTokenizer \
               --vocab-file $VOCAB_FILE \
               --epochs 3 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 32 \
               --hidden-size 2560 \
               --num-attention-heads 32 \
               --micro-batch-size 16 \
               --lr 5.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 512 \
               --max-position-embeddings 2050 \
               --save-interval 500000 \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --fp16 \
               --is-pipeline-compress False \
               --pipeline-compress-method ae \
               --pipeline-ae-dim 50 \
               --pipeline-qr-r 10 \
               --pipeline-k 50000 \
               --pipeline-m 50 \
               --pipeline-bits 8 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress False \
               --tensor-compress-method ae \
               --tensor-ae-dim 50 \
               --tensor-qr-r 10 \
               --tensor-k 50000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 16 \

