python preprocess_merge_data_single.py \
       --input ../aggregate \
       --output-prefix my-bert-agg \
       --vocab ../bert-large-cased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences