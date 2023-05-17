#!/bin/bash

#SBATCH --job-name=bert_base_pretrain    # create a short name for your job
#SBATCH --output=results/bert_base_pretrain.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=960:00:00          # total run time limit (HH:MM:SS)

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=../my-bert-wiki-and-book_text_sentence
VOCAB_FILE=../bert-base-cased-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_base_pretrain

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../pretrain_bert.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 128 \
       --global-batch-size 1024 \
       --seq-length 128 \
       --max-position-embeddings 512 \
       --train-iters 300000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 969,30,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 0.00001 \
       --lr-decay-iters 1000000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.01 \
       --log-interval 10000 \
       --save-interval 50000 \
       --eval-interval 10000 \
       --eval-iters 100 \
       --fp16 \
       --is-pipeline-compress False \
       --pipeline-compress-method topk \
       --pipeline-ae-dim 1024 \
       --pipeline-qr-r 10 \
       --pipeline-k 10000 \
       --pipeline-m 50 \
       --is-tensor-compress False \
       --tensor-compress-method topk \
       --tensor-ae-dim 50 \
       --tensor-qr-r 10 \
       --tensor-k 10000 \
       --tensor-m 50 \
