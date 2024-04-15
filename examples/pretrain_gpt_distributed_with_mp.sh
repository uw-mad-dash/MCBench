#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE=../opt-3b-vocab.json
MERGE_FILE=../opt-3b-merges.txt
DATA_PATH=/users/Master/Megatron-Resource/data/my-gpt2_text_document
#PRETRAINED_CHECKPOINT=checkpoints/opt_3b/split_2t_2p
CHECKPOINT_PATH=checkpoints/opt_3b_ds

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 32 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --tokenizer-type OPTTokenizer \
       --micro-batch-size 8 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 2050 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --is-pipeline-compress True \
       --pipeline-compress-method quantize \
       --pipeline-ae-dim 256 \
       --pipeline-qr-r 256 \
       --pipeline-k 500000 \
       --pipeline-m 50 \
       --pipeline-bits 8 \
       --start-pipeline-compress-rank 0 \
       --is-tensor-compress True \
       --tensor-compress-method quantize \
       --tensor-ae-dim 256 \
       --tensor-qr-r 256 \
       --tensor-k 500000 \
       --tensor-m 50 \
       --tensor-bits 8 \
       --start-tensor-compress-layer 16 \
