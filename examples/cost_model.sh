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
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_2t_2p
PRETRAINED_CHECKPOINT=None
CHECKPOINT_PATH=checkpoints/bert_345m_cola

for nl in 1 2 4
do
  for hidden in 1024 2304 3072 4096 6144 8192 10240 12288 16384 20480 25600
  do
    for bs in 32 64 128 16
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 4 \
               --pipeline-model-parallel-size 1 \
               --task CoLA \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs 1 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers $nl \
               --hidden-size $hidden \
               --num-attention-heads 16 \
               --micro-batch-size $bs \
               --lr 5.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 128 \
               --max-position-embeddings 512 \
               --save-interval 500000 \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --is-pipeline-compress False \
               --pipeline-compress-method ae \
               --pipeline-ae-dim 50 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method ae \
               --tensor-ae-dim 100 \
               --start-tensor-compress-layer 0
    done
  done
done

for nl in 1 2 4
do
  for hidden in 1024 2304 3072 4096 6144 8192 10240 12288 16384 20480 25600
  do
    for bs in 32 64 128 16
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 4 \
               --pipeline-model-parallel-size 1 \
               --task CoLA \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs 1 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers $nl \
               --hidden-size $hidden \
               --num-attention-heads 16 \
               --micro-batch-size $bs \
               --lr 5.0e-5 \
               --lr-warmup-fraction 0.065 \
               --seq-length 128 \
               --max-position-embeddings 512 \
               --save-interval 500000 \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --is-pipeline-compress False \
               --pipeline-compress-method ae \
               --pipeline-ae-dim 50 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress False \
               --tensor-compress-method ae \
               --tensor-ae-dim 50 \
               --start-tensor-compress-layer 0
    done
  done
done

#for nl in 1 2 4
#do
#  for hidden in 1024 2304 3072 4096 6144 8192 10240 12288 16384 20480 25600
#  do
#    for bs in 32 64 128 16
#    do
#      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
#               --tensor-model-parallel-size 4 \
#               --pipeline-model-parallel-size 1 \
#               --task CoLA \
#               --seed 1234 \
#               --train-data $TRAIN_DATA \
#               --valid-data $VALID_DATA \
#               --tokenizer-type BertWordPieceLowerCase \
#               --vocab-file $VOCAB_FILE \
#               --epochs 1 \
#               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
#               --num-layers $nl \
#               --hidden-size $hidden \
#               --num-attention-heads 16 \
#               --micro-batch-size $bs \
#               --lr 5.0e-5 \
#               --lr-warmup-fraction 0.065 \
#               --seq-length 128 \
#               --max-position-embeddings 512 \
#               --save-interval 500000 \
#               --log-interval 10 \
#               --eval-interval 100 \
#               --eval-iters 50 \
#               --weight-decay 1.0e-1 \
#               --is-pipeline-compress False \
#               --pipeline-compress-method ae \
#               --pipeline-ae-dim 50 \
#               --start-pipeline-compress-rank 0 \
#               --is-tensor-compress True \
#               --tensor-compress-method ae \
#               --tensor-ae-dim 50 \
#               --start-tensor-compress-layer 0
#    done
#  done
#done