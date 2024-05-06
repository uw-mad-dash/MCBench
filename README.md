# MCBench

This repository contains the data and code for paper [Does Compressing Activations Help Model Parallel Training? (MLSys'24)](https://arxiv.org/pdf/2301.02654). Our code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) developed by NVIDIA.

<p align="center">
    <a href="#installation">Installation 🛠️</a> •
    <a href="#data">Data 🗃️</a> •
    <a href="#checkpoint">Checkpoint ⚙️</a> •
    <a href="#quick-start">Quick Start 🚀</a> •
    <a href="#contributing">Contributing 🐜</a> •
</p>

## Installation

To get started, please first setup the environment:
```bash
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```
We employ Python 3.9 and CUDA 11.3. If you're using different versions of Python and CUDA, please ensure compatibility during the torch installation process.

To install apex, please proceed with the following steps:
```bash
git clone https://github.com/NVIDIA/apex.git
git checkout 22.04-dev
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data

We provide two examples illustrating how to prepare data for fine-tuning and pre-training, respectively.

### Fine-tuning

Download GLUE dataset:
```bash
python download_glue_data.py
```

Download vocabulary files:
```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt
```

### Pre-training

Download wikipedia dataset
```bash
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Preprocess wikipedia dataset
```bash
python -m wikiextractor.WikiExtractor -o output --json enwiki-latest-pages-articles.xml.bz2
cd tools
bash preprocess_wiki.sh
```

Download vocabulary files:
```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

## Checkpoint

Download Checkpoints:
```bash
cd examples
mkdir checkpoints
cd checkpoints
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
unzip megatron_bert_345m_v0.1_cased.zip -d bert_345m
```

Split the checkpoints:
```bash
cd tools
bash split_single.sh
```
Note: we need to set pipeline parallelism degree and tensor parallelism degree to fit the fine-tuning process.

## Quick Start

In the above section, the checkpoint is manually split with success. Here, we finetune BERT-345M (BERT-Large):
```bash
cd examples
bash finetune_mrpc_distributed_with_mp.sh
```

### Integration with Huggingface

To utilize the checkpoints from Huggingface, proceed with these steps:
1. Implement Transformer-based Model by using Transformer function provided by Megatron-LM.
2. Download checkpoints and preprocess the Huggingface checkpoints.
3. Split the checkpoints for fine-tuning.

Here, we present an example. Given that the BERT-Base model is already implemented in our repository, we will only demonstrate the final two steps.

Download and preprocess the Huggingface checkpoints
```bash
python preprocess_hf_bert_checkpoint.py
```

Split the checkpoints
```bash
bash split_single_hf.sh
```

Finetune BERT-Base:
```bash
cd examples
bash finetune_mrpc_bert_base_with_mp.sh
```

Expanding our repository to include additional Huggingface models requires us to independently implement these models. Here are several steps:
1. Implement parallel MLP and parallel Attention (please refer to `megatron/model/transformer.py`)
2. Implement the language model by using parallel MLP and parallel Attention (please refer to `megatron/model/language_model.py`)
3. Implement the model by using the above language model with embedding and head. (please refer to `megatron/model/bert_model.py` or `megatron/model/gpt_model.py`)

## Contributing

Authors: Song Bian*, Dacheng Li*, Hongyi Wang, Eric P. Xing, Shivaram Venkataraman

Affiliated: University of Wisconsin-Madison, Carnegie Mellon University, MBZUAI, and Petuum Inc.
