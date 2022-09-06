python preprocess_merge_data_single.py \
       --input ../openwebtext \
       --output-prefix my-openwebtext \
       --vocab ../bert-large-cased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences