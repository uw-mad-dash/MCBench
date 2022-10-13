#!/bin/bash

# compress method in [ae, quantize, topk, randk, topk_feedback, randk_feedback, qr]

#SBATCH --job-name=ft_qqp    # create a short name for your job
#SBATCH --output=results/1_4_qqp_32_512_topk_1000000.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../glue_data/QQP/train.tsv"
VALID_DATA="../glue_data/QQP/dev.tsv"
VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_4p
#PRETRAINED_CHECKPOINT=checkpoints/bert_book_pretrain_1000000_4_256
CHECKPOINT_PATH=checkpoints/bert_345m_qqp

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
               --tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 4 \
               --task QQP \
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
               --is-pipeline-compress True \
               --pipeline-compress-method topk \
               --pipeline-ae-dim 100 \
               --pipeline-qr-r 30 \
               --pipeline-k 1000000 \
               --pipeline-m 50 \
               --pipeline-bits 4 \
               --start-pipeline-compress-rank 1 \
               --is-tensor-compress False \
               --tensor-compress-method ae \
               --tensor-ae-dim 100 \
               --tensor-qr-r 30 \
               --tensor-k 10000 \
               --tensor-m 50 \
               --tensor-bits 4 \
               --start-tensor-compress-layer 12 \
