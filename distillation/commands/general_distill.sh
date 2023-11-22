#!/bin/bash
GPU=${1:-"0"}
TRAINING_TYPE=${2:-"general"}
# bash commands/general_distill.sh 0 general

max_train_steps=400000
weight_decay=0.01
batch_size=128
seq_len=128
gradient_accumulation_steps=2
learning_rate=6e-4
warmup=0.01
checkpoint_interval=50000
continue_distill=no

student_config=/path/minilm-6-384/config.json
teacher_path=/path/roberta-base
data_file=/path/wikibook.pickle
dump_path=dump/exp_general


export CUDA_VISIBLE_DEVICES=${GPU}
python train.py \
    --student_type roberta \
    --student_config  ${student_config} \
    --teacher_type roberta \
    --teacher_name roberta-base \
    --teacher_path  ${teacher_path} \
    --data_file ${data_file} \
    --dump_path ${dump_path} \
    --weight_decay ${weight_decay} \
    --max_train_steps ${max_train_steps} \
    --batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_prop ${warmup} \
    --learning_rate ${learning_rate} \
    --max_seq_len ${seq_len} \
    --continue_distill ${continue_distill} \
    --teacher_size base \
    --checkpoint_interval ${checkpoint_interval} \
    --force \
    --use_lengthdrop \
    --training_type ${TRAINING_TYPE} \
    --fp16 \