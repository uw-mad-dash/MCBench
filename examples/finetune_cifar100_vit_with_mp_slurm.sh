#!/bin/bash

# compress method in [ae, quantize, topk_int, randk_int, topk, randk, topk_feedback, randk_feedback, qr]

#SBATCH --job-name=ft_cifar100    # create a short name for your job
#SBATCH --output=results_vit/2_2_cifar100_ae_100_lr_0.1.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

PRETRAINED_CHECKPOINT=checkpoints/vision_classify_large_patch16/split_2t_2p

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/vision/main.py \
               --is-vision-train True \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --task classify \
               --dataset-name 'cifar100' \
               --num-classes 100 \
               --epochs 100 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --patch-size 16 \
               --encoder-seq-length 512 \
               --decoder-seq-length 512 \
               --micro-batch-size 64 \
               --max-position-embeddings 512 \
               --optimizer 'sgd' \
               --lr 0.01 \
               --lr-decay-style 'cosine' \
               --weight-decay 0.0 \
               --save-interval 500000 \
               --log-interval 100 \
               --eval-interval 1000 \
               --eval-iters 100 \
               --layernorm-epsilon 1e-12 \
               --hidden-dropout 0.0 \
               --attention-dropout 0.0 \
               --is-pipeline-compress True \
               --pipeline-compress-method ae \
               --pipeline-ae-dim 100 \
               --pipeline-qr-r 10 \
               --pipeline-k 100 \
               --pipeline-m 50 \
               --pipeline-bits 8 \
               --start-pipeline-compress-rank 0 \
               --is-tensor-compress True \
               --tensor-compress-method ae \
               --tensor-ae-dim 100 \
               --tensor-qr-r 10 \
               --tensor-k 80000 \
               --tensor-m 50 \
               --tensor-bits 8 \
               --start-tensor-compress-layer 12 \
