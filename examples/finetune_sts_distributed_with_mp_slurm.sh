#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

#SBATCH --job-name=ft_sts    # create a short name for your job
#SBATCH --output=results/4_1_sts_pretrain_topk_int_1600000.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/STS-B/train.tsv"
VALID_DATA="../glue_data/STS-B/dev.tsv"
VOCAB_FILE="../bert-large-cased-vocab.txt"
#PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_4p
PRETRAINED_CHECKPOINT=checkpoints/4_1_bert_book_pretrain_topk_int_1600000
CHECKPOINT_PATH=checkpoints/bert_345m_sts

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 4 \
               --pipeline-model-parallel-size 1 \
               --task STS \
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
               --pipeline-compress-method topk_int \
               --pipeline-ae-dim 100 \
               --pipeline-qr-r 30 \
               --pipeline-k 80000 \
               --pipeline-m 50 \
               --pipeline-bits 8 \
               --start-pipeline-compress-rank 1 \
               --is-tensor-compress False \
               --tensor-compress-method quantize \
               --tensor-ae-dim 50 \
               --tensor-qr-r 30 \
               --tensor-k 200000 \
               --tensor-m 50 \
               --tensor-bits 2 \
               --start-tensor-compress-layer 12 \
