#!/bin/bash

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="/home/wisr/song/Megatron-LM/RACE/train/middle"
VALID_DATA="/home/wisr/song/Megatron-LM/RACE/dev/middle \
            /home/wisr/song/Megatron-LM/RACE/dev/high"
VOCAB_FILE="/home/wisr/song/Megatron-LM/bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=../bert_cased/release/mp_rank_00/model_optim_rng.pt
CHECKPOINT_PATH=checkpoints/bert_345m_race

python -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --sequence-parallel \
               --task RACE \
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
               --micro-batch-size 4 \
               --lr 1.0e-5 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.06 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --save-interval 100000 \
               --save $CHECKPOINT_PATH \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --clip-grad 1.0 \
               --hidden-dropout 0.1 \
               --attention-dropout 0.1 \
               --fp16 \
               --is-pipeline-compress False \
               --pipeline-compress-dim 2 \
               --is-tensor-compress False \
               --tensor-compress-dim 2