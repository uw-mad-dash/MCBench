#!/bin/bash

# compress method in [ae, quantize, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes $WORLD_SIZE \
                  --node_rank $1 \
                  --master_addr 44.205.3.63 \
                  --master_port 6001"

TRAIN_DATA="../glue_data/MRPC/msr_paraphrase_train.txt"
VALID_DATA="../glue_data/MRPC/msr_paraphrase_test.txt"
VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_4t
CHECKPOINT_PATH=checkpoints/bert_345m_mrpc

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 4 \
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
               --micro-batch-size 32 \
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
               --pipeline-ae-dim 50 \
               --pipeline-qr-r 10 \
               --pipeline-k 10000 \
               --pipeline-m 50 \
               --is-tensor-compress True \
               --tensor-compress-method ae \
               --tensor-ae-dim 50 \
               --tensor-qr-r 10 \
               --tensor-k 10000 \
               --tensor-m 50 \
#               --no-masked-softmax-fusion # My environment (orca) can't handle this, feel free to comment it in AWS