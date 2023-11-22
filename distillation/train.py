# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training the distilled model.
"""
import argparse
import os, sys
import pickle
import shutil
import json
import torch
from distiller import Distiller

from modeling.modeling_roberta import RobertaForTokenPruning

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
)
from utils import init_gpu_params, logger, set_seed

from customized_criterions.lad import LAD
from customized_datasets.lm_seqs_dataset import LmSeqsDataset

MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaForTokenPruning, RobertaTokenizer),
}


def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--force", action="store_true", help="Overwrite dump_path if it already exists.")

    parser.add_argument(
        "--dump_path", type=str, required=True, help="The output directory (log, checkpoints, parameters, etc.)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="The binarized file (tokenized + tokens_to_ids) and grouped by sequence.",
    )

    parser.add_argument(
        "--student_type",
        type=str,
        choices=["distilbert", "roberta", "gpt2", 'bert'],
        required=True,
        help="The student type (DistilBERT, RoBERTa).",
    )
    parser.add_argument("--student_config", type=str, required=True, help="Path to the student configuration.")
    parser.add_argument(
        "--student_pretrained_weights", default=None, type=str, help="Load student initialization checkpoint."
    )

    parser.add_argument(
        "--teacher_type", choices=["bert", "roberta", "gpt2"], required=True, help="Teacher type (BERT, RoBERTa)."
    )
    parser.add_argument("--teacher_name", type=str, required=True, help="The teacher model.")

    parser.add_argument(
        "--alpha_ce", default=0.5, type=float, help="Linear weight for the distillation loss. Must be >=0."
    )
    parser.add_argument(
        "--alpha_mlm",
        default=0.0,
        type=float,
        help="Linear weight for the MLM loss. Must be >=0. Should be used in coonjunction with `mlm` flag.",
    )
    parser.add_argument("--alpha_clm", default=0.5, type=float, help="Linear weight for the CLM loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.0, type=float, help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument(
        "--alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0."
    )

    parser.add_argument(
        "--mlm", action="store_true", help="The LM step: MLM or CLM. If `mlm` is True, the MLM is used over CLM."
    )
    parser.add_argument(
        "--mlm_mask_prop",
        default=0.15,
        type=float,
        help="Proportion of tokens for which we need to make a prediction.",
    )
    parser.add_argument("--word_mask", default=0.8, type=float, help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float, help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float, help="Proportion of tokens to randomly replace.")
    parser.add_argument(
        "--mlm_smoothing",
        default=0.7,
        type=float,
        help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).",
    )
    parser.add_argument("--token_counts", type=str, help="The token counts in the data_file for MLM.")

    parser.add_argument(
        "--restrict_ce_to_mask",
        action="store_true",
        help="If true, compute the distilation loss only the [MLM] prediction distribution.",
    )
    parser.add_argument(
        "--freeze_pos_embs",
        action="store_true",
        help="Freeze positional embeddings during distillation. For student_type in ['roberta', 'gpt2'] only.",
    )
    parser.add_argument(
        "--freeze_token_type_embds",
        action="store_true",
        help="Freeze token type embeddings during distillation if existent. For student_type in ['roberta'] only.",
    )

    parser.add_argument("--max_seq_len", type=int, default=128, help="Number of pass on the whole dataset.")
    parser.add_argument("--n_epoch", type=int, default=3, help="Number of pass on the whole dataset.")
    parser.add_argument("--max_train_steps", type=int, default=100000, help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (for each process).")
    parser.add_argument(
        "--group_by_size",
        action="store_false",
        help="If true, group sequences that have similar length into the same batch. Default is true.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=50,
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument("--warmup_prop", default=0.05, type=float, help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    parser.add_argument("--log_interval", type=int, default=50000000, help="Tensorboard logging interval.")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=50000, help="Checkpoint interval.")
    parser.add_argument("--teacher_path", type=str, required=True, help="The teacher model.")

    parser.add_argument("--n_relation_heads", type=int, default=64, required=False)
    parser.add_argument("--continue_distill", type=str, choices=['yes', 'no'], required=True)

    parser.add_argument("--teacher_size", type=str, default='large',
                        choices=['base', 'large'], required=False)
    parser.add_argument("--temperature", default=1.0, type=float, help="")
    parser.add_argument("--use_lengthdrop", action="store_true")
    parser.add_argument("--lengthdropratio", default=0.2, type=float, help="")
    parser.add_argument("--training_type", default='general', type=str, choices=['general', 'task', 'taskda'], help="")
    args = parser.parse_args()

    # ARGS #
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        if os.path.exists(args.dump_path):
            if not args.force:
                raise ValueError(
                    f"Serialization dir {args.dump_path} already exists, but you have not precised wheter to overwrite it"
                    "Use `--force` if you want to overwrite it"
                )
            else:
                shutil.rmtree(args.dump_path)

        if not os.path.exists(args.dump_path):
            os.makedirs(args.dump_path)
        logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

        # SAVE PARAMS #
        logger.info(f"Param: {args}")
        with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    student_config_class, student_model_class, _ = MODEL_CLASSES[args.student_type]
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES[args.teacher_type]
    # TOKENIZER #
    tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_path)
    special_tok_ids = {}
    for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
        idx = tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    logger.info(f"Special tokens {special_tok_ids}")
    args.special_tok_ids = special_tok_ids
    args.max_model_input_size = tokenizer.max_model_input_sizes[args.teacher_name]

    # DATA LOADER #
    train_taskda_dataset = None
    logger.info(f"Loading data from {args.data_file}")
    with open(args.data_file, "rb") as fp:
        data_train = pickle.load(fp)
    train_dataset = LmSeqsDataset(params=args, data=data_train)
    eval_dataset = None

    # TEACHER #
    if args.teacher_path != '':
        teacher = teacher_model_class.from_pretrained(args.teacher_path, is_teacher=True,
                                                      teacher_size=args.teacher_size,
                                                      classification=False)
        logger.info(f"Teacher loaded from {args.teacher_path}.")
    else:
        teacher = teacher_model_class.from_pretrained(args.teacher_name, is_teacher=True,
                                                      teacher_size=args.teacher_size,
                                                      classification=False)
        logger.info(f"Teacher loaded from {args.teacher_name}.")

    # STUDENT #
    logger.info(f"Loading student config from {args.student_config}")
    stu_architecture_config = student_config_class.from_pretrained(args.student_config)
    stu_architecture_config.output_hidden_states = True

    if args.student_pretrained_weights is not None and args.continue_distill == 'yes':
        logger.info(f"Loading pretrained weights from {args.student_pretrained_weights}")
        student = student_model_class.from_pretrained(args.student_pretrained_weights, config=stu_architecture_config,
                                                      is_teacher=False,
                                                      classification=False,
                                                      use_lengthdrop=args.use_lengthdrop)
    else:
        logger.info(f"Randomly initializing student model.")
        student = student_model_class(config=stu_architecture_config,
                                      is_teacher=False,
                                      classification=False,
                                      use_lengthdrop=args.use_lengthdrop)

    if args.n_gpu > 0:
        student.to(f"cuda:{args.local_rank}")
    logger.info("Student loaded.")

    if args.n_gpu > 0:
        teacher.to(f"cuda:{args.local_rank}")

    # SANITY CHECKS #
    assert student.config.vocab_size == teacher.config.vocab_size
    assert student.config.max_position_embeddings == teacher.config.max_position_embeddings


    criterion = LAD(student, dim=768, special_tok_ids=args.special_tok_ids, teacher_model=teacher)

    if args.n_gpu > 0:
        criterion.to(f"cuda:{args.local_rank}")

    logger.info("Distiller")
    # DISTILLER #
    torch.cuda.empty_cache()
    distiller = Distiller(
        params=args, dataset=train_dataset, student=student, teacher=teacher, criterion=criterion)
    distiller.train()
    logger.info("Let's go get some drinks.")


if __name__ == "__main__":
    main()