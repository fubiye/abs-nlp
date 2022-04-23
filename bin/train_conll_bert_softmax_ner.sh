#!/bin/bash
DATASET='conll2003'
MODEL_TYPE='bert'
MODEL_NAME_OR_PATH='bert-base-uncased'
OUTPUT_DIR='output/bert'

python train.py \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--dataset $DATASET \
--do_train \
--do_eval \
--evaluate_during_training \
--adv_training fgm \
--num_train_epochs 3 \
--max_seq_length 128 \
--logging_steps 0.2 \
--batch_size 16 \
--learning_rate 5e-5 \
--bert_lr 5e-5 \
--classifier_lr  5e-5 \
