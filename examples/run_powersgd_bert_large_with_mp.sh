#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_2t_2p

for method in 'power'
#'ef_power'
do
  for r in 50 100
  do
    for dataset in 'cola' 'mnli' 'qqp' 'sst' 'mrpc' 'qnli' 'rte' 'sts'
# 'cola' 'mnli' 'qqp' 'sst' 'mrpc' 'qnli' 'rte' 'sts'
    do
      if [ $dataset == 'mnli' ];
      then
        TRAIN_DATA="../glue_data/MNLI/train.tsv"
        VALID_DATA="../glue_data/MNLI/dev_matched.tsv \
                    ../glue_data/MNLI/dev_mismatched.tsv"
        TASK="MNLI"
        EPOCH=3
      elif [ $dataset == 'qqp' ];
      then
        TRAIN_DATA="../glue_data/QQP/train.tsv"
        VALID_DATA="../glue_data/QQP/dev.tsv"
        TASK="QQP"
        EPOCH=3
      elif [ $dataset == 'sst' ];
      then
        TRAIN_DATA="../glue_data/SST-2/train.tsv"
        VALID_DATA="../glue_data/SST-2/dev.tsv"
        TASK="SST"
        EPOCH=3
      elif [ $dataset == 'mrpc' ];
      then
        TRAIN_DATA="../glue_data/MRPC/msr_paraphrase_train.txt"
        VALID_DATA="../glue_data/MRPC/msr_paraphrase_test.txt"
        TASK="MRPC"
        EPOCH=5
      elif [ $dataset == 'cola' ];
      then
        TRAIN_DATA="../glue_data/CoLA/train.tsv"
        VALID_DATA="../glue_data/CoLA/dev.tsv"
        TASK="CoLA"
        EPOCH=3
      elif [ $dataset == 'qnli' ];
      then
        TRAIN_DATA="../glue_data/QNLI/train.tsv"
        VALID_DATA="../glue_data/QNLI/dev.tsv"
        TASK="QNLI"
        EPOCH=3
      elif [ $dataset == 'rte' ];
      then
        TRAIN_DATA="../glue_data/RTE/train.tsv"
        VALID_DATA="../glue_data/RTE/dev.tsv"
        TASK="RTE"
        EPOCH=3
      elif [ $dataset == 'sts' ];
      then
        TRAIN_DATA="../glue_data/STS-B/train.tsv"
        VALID_DATA="../glue_data/STS-B/dev.tsv"
        TASK="STS"
        EPOCH=3
      fi
      python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task $TASK \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs $EPOCH \
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
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --fp16 \
               --is-pipeline-compress True \
               --pipeline-compress-method $method \
               --pipeline-qr-r $r \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method $method \
               --tensor-qr-r $r \
               --start-tensor-compress-layer 12
    done
  done
done


