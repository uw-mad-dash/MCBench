#!/bin/bash


WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes $WORLD_SIZE \
                  --node_rank $1 \
                  --master_addr 10.117.1.38 \
                  --master_port 6000"

TRAIN_DATA="../glue_data/MRPC/msr_paraphrase_train.txt"
VALID_DATA="../glue_data/MRPC/msr_paraphrase_test.txt"
VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_16
CHECKPOINT_PATH=checkpoints/bert_345m_mrpc

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size $WORLD_SIZE \
               --pipeline-model-parallel-size 1 \
               --task MRPC \
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
               --is-pipeline-compress False \
               --pipeline-compress-dim 1024 \
               --is-tensor-compress False \
               --tensor-compress-dim 1024 \
               --is-quantize True \
               --no-masked-softmax-fusion # My environment (orca) can't handle this, feel free to comment it in AWS
