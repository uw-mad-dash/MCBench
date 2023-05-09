#!/bin/bash

#SBATCH --job-name=split    # create a short name for your job
#SBATCH --output=split.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4              # number of gpus per node
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)

TARGET_PIPELINE_MODEL_PARALLEL_SIZE=2
TARGET_TENSOR_MODEL_PARALLEL_SIZE=2

CHECKPOINT_PATH=../examples/checkpoints/vision_classify_large_patch32

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


python3 -m torch.distributed.launch $DISTRIBUTED_ARGS split_mp_partitions_single_vit_hf.py \
        --model-type 'vit' \
        --vision-backbone-type 'vit' \
        --task classify \
        --finetune \
        --num-classes 10 \
        --dataset-name "cifar10" \
        --pipeline-model-parallel-size 2 \
        --tensor-model-parallel-size 2 \
        --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
        --target-tensor-model-parallel-size $TARGET_TENSOR_MODEL_PARALLEL_SIZE \
        --encoder-seq-length 512 \
        --decoder-seq-length 512 \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --patch-size 32 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH/split_2t_2p \
        --micro-batch-size 1 \
        --save-interval 500000