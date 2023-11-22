#!/bin/bash
GPU=${1:-"0"}
TRAINING_TYPE=${2:-"task"}
TASK_NAME=${3:-"mnli"}

# bash commands/task_distill.sh 0 task mnli

batch_size=32
gradient_accumulation_steps=1
seq_len=128
warmup=0.1
weight_decay=0.01
continue_distill=yes

student_config=/path/minilm-6-384/config.json
student_pretrained_weights=/path/minilm-6-384-generaldistillation
teacher_path=/path/roberta-base-${TASK_NAME}
data_file=/path/${TASK_NAME}/task.pickle
dump_path=dump/exp_task_${TASK_NAME}



if [ "${TASK_NAME}" = "mnli" ]; then
  learning_rate=5e-5
  max_train_steps=245439
  checkpoint_interval=50000
elif [ "${TASK_NAME}" = "mrpc" ]; then
  learning_rate=1e-5 # best 这个应该是最佳的
  max_train_steps=2293
  checkpoint_interval=1000
elif [ "${TASK_NAME}" = "rte" ]; then
  learning_rate=1e-5
  max_train_steps=1556
  checkpoint_interval=700
elif [ "${TASK_NAME}" = "qnli" ]; then
  learning_rate=3e-5
  max_train_steps=65465
  checkpoint_interval=10000
elif [ "${TASK_NAME}" = "sst2" ]; then
  learning_rate=5e-5
  max_train_steps=42094
  checkpoint_interval=10000
elif [ "${TASK_NAME}" = "cola" ]; then
  learning_rate=3e-5
  max_train_steps=13360
  checkpoint_interval=3000
elif [ "${TASK_NAME}" = "qqp" ]; then
  learning_rate=7e-5
  max_train_steps=227404
  checkpoint_interval=50000
elif [ "${TASK_NAME}" = "stsb" ]; then
  learning_rate=3e-5
  max_train_steps=3594
  checkpoint_interval=1000
fi

export CUDA_VISIBLE_DEVICES=${GPU}
python train.py \
    --student_type roberta \
    --student_config  ${student_config} \
    --student_pretrained_weights ${student_pretrained_weights} \
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