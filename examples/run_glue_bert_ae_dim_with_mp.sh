#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/QNLI/train.tsv"
VALID_DATA="../glue_data/QNLI/dev.tsv"
VOCAB_FILE="../bert-large-cased-vocab.txt"
#PRETRAINED_CHECKPOINT=checkpoints/bert_base_hf/split_2t_2p
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_2t_2p

for dim in 1 5 10 20 50 100 200 400 800 1024
do
  python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
           --tensor-model-parallel-size 2 \
           --pipeline-model-parallel-size 2 \
           --task QNLI \
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
           --micro-batch-size 32 \
           --lr 5.0e-5 \
           --lr-warmup-fraction 0.065 \
           --seq-length 128 \
           --max-position-embeddings 512 \
           --save-interval 500000 \
           --log-interval 10 \
           --eval-interval 100 \
           --eval-iters 50 \
           --weight-decay 1.0e-1 \
           --fp16 \
           --is-pipeline-compress True \
           --pipeline-compress-method ae \
           --pipeline-ae-dim $dim \
           --start-pipeline-compress-rank 0 \
           --is-tensor-compress True \
           --tensor-compress-method ae \
           --tensor-ae-dim $dim \
           --start-tensor-compress-layer 12
done
