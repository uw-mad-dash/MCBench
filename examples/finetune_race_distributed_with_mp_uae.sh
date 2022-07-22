#!/bin/bash

#SBATCH --job-name=ft_race    # create a short name for your job
#SBATCH --output=results/4_1_race_randk_1000.txt
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

TRAIN_DATA="../RACE/train/middle \
            ../RACE/train/high"
VALID_DATA="../RACE/test/middle \
            ../RACE/test/high"
VOCAB_FILE="../bert-large-cased-vocab.txt"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m/split_16
CHECKPOINT_PATH=checkpoints/bert_345m_race

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
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
               --micro-batch-size 8 \
               --lr 2.0e-5 \
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
               --pipeline-compress-method ae \
               --pipeline-ae-dim 1024 \
               --pipeline-qr-r 10 \
               --pipeline-k 1000 \
               --is-tensor-compress True \
               --tensor-compress-method quantize \
               --tensor-ae-dim 100 \
               --tensor-qr-r 10 \
               --tensor-k 1000 \
