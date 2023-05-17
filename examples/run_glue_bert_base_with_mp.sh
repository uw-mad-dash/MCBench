#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6001"

TRAIN_DATA="../glue_data/QNLI/train.tsv"
VALID_DATA="../glue_data/QNLI/dev.tsv"
VOCAB_FILE="../bert-base-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_base_hf/split_2t_2p

for method in 'ae'
do
  if [ $method == 'ae' ];
  then
    for dim in 50 100
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main_hf.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task QNLI \
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
               --pipeline-compress-method $method \
               --pipeline-ae-dim $dim \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-ae-dim $dim \
               --start-tensor-compress-layer 6
    done
  elif [ $method == 'quantize' ];
  then
    for bits in 2 4 8
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main_hf.py \
               --is-skip-compression-inference True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task STS \
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
               --pipeline-compress-method $method \
               --pipeline-bits $bits \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-bits $bits \
               --start-tensor-compress-layer 6
    done
  elif [ $method == 'topk_int' ];
  then
    for k in 40000 80000 200000 400000
    do
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main_hf.py \
               --is-skip-compression-inference True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task STS \
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
               --pipeline-compress-method $method \
               --pipeline-k $k \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-k $k \
               --start-tensor-compress-layer 6
    done
  fi
done


