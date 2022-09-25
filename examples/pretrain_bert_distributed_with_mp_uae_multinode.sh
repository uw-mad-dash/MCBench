#!/bin/bash

#SBATCH --job-name=4_4_dacheng_bert_pretrain    # create a short name for your job
#SBATCH --output=results/4_4_dacheng_bert_pretrain.txt
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=960:00:00          # total run time limit (HH:MM:SS)

export MASTER_PORT=12311
export WORLD_SIZE=16
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
#
#export GLOO_SOCKET_IFNAME=eno1
#export NCCL_SOCKET_IFNAME=eno1

DATA_PATH=../my-bert-wiki-and-book_text_sentence
VOCAB_FILE=../bert-large-cased-vocab.txt
CHECKPOINT_PATH=checkpoints/4_4_dacheng_bert_pretrain

options="\
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 256 \
       --global-batch-size 1024 \
       --seq-length 128 \
       --max-position-embeddings 512 \
       --train-iters 500000 \
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
       --lr-decay-iters 500000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.01 \
       --log-interval 5000 \
       --save-interval 5000 \
       --eval-interval 5000 \
       --eval-iters 100 \
       --fp16 \
       --is-pipeline-compress False \
       --pipeline-compress-method topk \
       --pipeline-ae-dim 1024 \
       --pipeline-qr-r 10 \
       --pipeline-k 10000 \
       --pipeline-m 50 \
       --pipeline-bits 8 \
       --is-tensor-compress False \
       --tensor-compress-method topk \
       --tensor-ae-dim 50 \
       --tensor-qr-r 10 \
       --tensor-k 10000 \
       --tensor-m 50 \
       --tensor-bits 8 \
       --multinode-train True "

run_cmd="python -u ../pretrain_bert.py $@ ${options}"

srun -l \
     --output=results/4_4_dacheng_bert_pretrain.txt sh -c "${run_cmd}"
