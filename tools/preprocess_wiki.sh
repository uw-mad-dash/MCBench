python preprocess_data_wiki.py \
       --input ../data/my-corpus.json \
       --output-prefix ../data/my-bert \
       --vocab-file ../bert-large-uncased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers 32 \
       --chunk-size 1