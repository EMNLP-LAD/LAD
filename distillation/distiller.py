# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import math
import os
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from transformers import get_linear_schedule_with_warmup
from utils import logger
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class LengthDropArguments:
    length_config: Optional[List[int]] = None
    length_adaptive: Optional[bool] = False
    num_sandwich: Optional[int] = 2
    length_drop_ratio_bound: Optional[float] = 0.2
    layer_dropout_prob: Optional[float] = 0.2
    layer_dropout_bound: Optional[int] = 0


class Distiller:
    def __init__(
        self, params: dict, dataset, student, teacher, criterion):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.training_type = params.training_type

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size
        self.task_name = self.params.data_file.split('/')[-2]

        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.global_steps = 0
        self.retrieval_steps = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_log = 0

        self.max_train_steps = params.max_train_steps
        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1

        # fixme
        if params.max_train_steps is None:
            self.num_steps_epoch = len(self.dataloader)
            num_train_optimization_steps = (
                int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
            )
        else:
            num_train_optimization_steps = params.max_train_steps
            self.num_train_optimization_steps = num_train_optimization_steps
            logger.info("self.num_train_optimization_steps: %d" % self.num_train_optimization_steps)


        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=False,    # fixme fp32
                )

        self.is_master = params.is_master


    def prepare_batch(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.
        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.
        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        # true false

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size
        return token_ids, attn_mask

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            # if self.mlm:
            #     pad_id = self.params.special_tok_ids["pad_token"]
            # else:
            #     pad_id = self.params.special_tok_ids["unk_token"]
            pad_id = self.params.special_tok_ids["pad_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        end_train = False
        while True:
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            best_metric = 0.
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)

                self.retrieval_steps += 1
                if self.retrieval_steps % self.params.gradient_accumulation_steps == 0:
                    self.global_steps += 1

                token_ids, attn_mask = self.prepare_batch(batch)
                self.step(input_ids=token_ids, attention_mask=attn_mask)
                iter_bar.update()
                iter_bar.set_postfix(
                    {"STEPs": f"{self.global_steps:d}",
                     "LR": f"{self.scheduler.get_lr()[0]:.6f}",
                     "Last_loss": f"{self.last_loss:.2f}",
                     "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )

                if self.global_steps % self.params.checkpoint_interval == 0:
                    if self.is_master:
                        self.save_checkpoint(checkpoint_name=f"model_step_{self.global_steps}.pth")

                if self.global_steps == self.num_train_optimization_steps:
                    end_train = True
                    break
            iter_bar.close()
            if end_train:
                logger.info(f"--- Ending TRAINING {self.global_steps}/{self.num_train_optimization_steps}")
                # if self.is_master:
                #     self.save_checkpoint(checkpoint_name=f"model_step_{self.num_train_optimization_steps}.pth")
                # break # fixme
                end_train = False
                if self.is_master:
                    logger.info("Save very last checkpoint as `pytorch_model.bin`.")
                    self.save_checkpoint(checkpoint_name="pytorch_model.bin")
                    logger.info("Training is finished")

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")

            self.end_epoch()

    def sampling_tokendrop_config(self, max_seq_length, model_config_numhiddenlayers, length_drop_ratio_bound):
        def sample_length_configuration(
                max_seq_length,
                num_hidden_layers,
                layer_config=None,
                length_drop_ratio_bound=None,
                min_length=2,
        ):
            length = max_seq_length
            length_configuration = ()
            for i in range(num_hidden_layers):
                if layer_config is None or i in layer_config:
                    length = np.random.randint(int(np.ceil(length * (1 - length_drop_ratio_bound))), length + 1)
                length = max(length, min_length)
                if i == (num_hidden_layers - 1):
                    length_configuration += (length_configuration[-1],)
                else:
                    length_configuration += (length,)
            return length_configuration

        length_config = sample_length_configuration(
            max_seq_length,
            model_config_numhiddenlayers,
            length_drop_ratio_bound=length_drop_ratio_bound,
        )
        return length_config


    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels=None):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        output_attentions = True
        output_hidden_states = True
        output_qkvs = True
        output_logits = False

        # sample length config
        if hasattr(self.student, "module"):
            model_config_numhiddenlayers = self.student.module.config.num_hidden_layers
        else:
            model_config_numhiddenlayers = self.student.config.num_hidden_layers
        max_seq_length = input_ids.shape[1]
        default_batchsize = self.params.batch_size

        if self.params.use_lengthdrop:
            if self.training_type == 'general':
                lengthdropratio = np.random.randint(1, 8) * 0.1
            else:
                lengthdropratio = np.random.randint(0, 8) * 0.1
            length_config = self.sampling_tokendrop_config(max_seq_length=max_seq_length,
                                                           model_config_numhiddenlayers=model_config_numhiddenlayers,
                                                           length_drop_ratio_bound=lengthdropratio)
        else:
            length_config = None

        with torch.no_grad():
            t_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     output_qkvs=output_qkvs,
                                     output_logits=output_logits)
        if self.training_type == 'general':
            s_outputs_ori = self.student(input_ids=input_ids, attention_mask=attention_mask,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                output_qkvs=output_qkvs,
                                length_config=None)
        else:
            s_outputs_ori = None
        s_outputs_drop = self.student(input_ids=input_ids, attention_mask=attention_mask,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     output_qkvs=output_qkvs,
                                     length_config=length_config)

        loss = self.criterion((s_outputs_ori, s_outputs_drop), t_outputs, attention_mask, input_ids, default_batchsize,
                              self.training_type)
        if loss is not None:
            self.total_loss_epoch += loss.item()
            self.last_loss = loss.item()
            self.optimize(loss)
            self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            # self.global_steps += 1

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")
        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 1
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
