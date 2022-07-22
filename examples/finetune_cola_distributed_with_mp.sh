#!/bin/bash

# compress method in [ae, quantize, topk, randk, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/RTE/train.tsv"
VALID_DATA="../glue_data/RTE/dev.tsv"
VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split
CHECKPOINT_PATH=checkpoints/bert_345m_cola

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task RTE \
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
               --is-pipeline-compress False \
               --pipeline-compress-method ae \
               --pipeline-ae-dim 1024 \
               --pipeline-qr-r 10 \
               --pipeline-k 1000 \
               --is-tensor-compress True \
               --tensor-compress-method quantize \
               --tensor-ae-dim 100 \
               --tensor-qr-r 10 \
               --tensor-k 1000 \

