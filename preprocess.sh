#!/bin/bash

#SBATCH --job-name=preprocess    # create a short name for your job
#SBATCH --output=examples/results/bert_dataset_preprocess.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=64       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=202:00:00         # total run time limit (HH:MM:SS)

python3 tools/preprocess_data_single.py \
       --input text \
       --output-prefix my-bert \
       --vocab bert-large-cased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
#       --workers 1 \
#       --chunk-size 200