#!/bin/bash
GPU=${1:-"0"}
TRAINING_TYPE=${2:-"taskda"}
TASK_NAME=${3:-"mnli"}

# bash commands/taskda_distill.sh 0 taskda mnli

batch_size=256
gradient_accumulation_steps=1
seq_len=128
warmup=0.06
weight_decay=0.01
continue_distill=yes

student_config=/path/minilm-6-384/config.json
student_pretrained_weights=/path/minilm-6-384-generaldistillation
teacher_path=/path/roberta-base-${TASK_NAME}
data_file=/path/${TASK_NAME}/taskda.pickle
dump_path=dump/exp_taskda_${TASK_NAME}


if [ "${TASK_NAME}" = "mnli" ]; then
  # 8017849 examples after data augmentation
  learning_rate=1e-4
  max_train_steps=313197
  checkpoint_interval=50000
elif [ "${TASK_NAME}" = "mrpc" ]; then
  # 225057 examples after data augmentation
  learning_rate=1e-4
  max_train_steps=17582
  checkpoint_interval=5000
elif [ "${TASK_NAME}" = "rte" ]; then
  # 143018 examples after data augmentation
  learning_rate=1e-4
  max_train_steps=11173
  checkpoint_interval=3000
elif [ "${TASK_NAME}" = "qnli" ]; then
  # 4229751 examples after data augmentation
  learning_rate=5e-5
  max_train_steps=165224
  checkpoint_interval=50000
elif [ "${TASK_NAME}" = "sst2" ]; then
  # 1107141 examples after data augmentation
  learning_rate=5e-5
  max_train_steps=86495
  checkpoint_interval=20000
elif [ "${TASK_NAME}" = "cola" ]; then
  # 210911 examples after data augmentation
  learning_rate=5e-5
  max_train_steps=41193
  checkpoint_interval=10000
elif [ "${TASK_NAME}" = "stsb" ]; then
  # 319959 examples after data augmentation
  learning_rate=5e-5
  max_train_steps=24996
  checkpoint_interval=10000
elif [ "${TASK_NAME}" = "qqp" ]; then
  # 7573531 examples after data augmentation
  learning_rate=1e-4
  max_train_steps=295842
  checkpoint_interval=50000
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