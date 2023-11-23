import torch
from torch import nn
import math
import torch.distributed as dist
eps = 1e-7
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class LAD(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, student_model, dim, K=16384, T=0.07, max_seq_len=128, special_tok_ids=None, teacher_model=None):
        super(LAD, self).__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.K = K
        self.K_sent = 16384
        self.T = T
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.loss_mse = MSELoss()
        self.mse_sum = MSELoss(reduction='sum')
        self.special_tok_ids = special_tok_ids
        print('dim: %d' % dim)
        print('negative k: %d' % K)
        print('negative sent k: %d' % self.K_sent)
        print('temperature tau: %f' % T)

        # create the encoders
        self.encoder_q = student_model

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K_sent))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue_token
        self.register_buffer("queue_token_last1", torch.randn(dim, self.K))
        self.queue_token_last1 = nn.functional.normalize(self.queue_token_last1, dim=0)
        self.register_buffer("queue_ptr_token_last1", torch.zeros(1, dtype=torch.long))

        # create the queue_token
        self.register_buffer("queue_token_last2", torch.randn(dim, self.K))
        self.queue_token_last2 = nn.functional.normalize(self.queue_token_last2, dim=0)
        self.register_buffer("queue_ptr_token_last2", torch.zeros(1, dtype=torch.long))

        self.deleted_num = 0
        self.visible_num = 0
        self.training_steps = 0

    def update_teacher_queue_last1(self, teacher, attention_mask):
        t = teacher['hidden_states'][-1][:, 1:, :]
        attention_mask = attention_mask[:, 1:]
        gather_index = torch.nonzero(attention_mask.reshape(-1) == 1).squeeze(1)  # batch x len --
        # print(attention_mask)
        # print(gather_index)
        # sys.exit()
        t = t.reshape(-1, t.shape[-1]).index_select(dim=0, index=gather_index).detach()  # batch x len --, dim
        k = nn.functional.normalize(t, dim=-1)  # batch x len --, dim
        self._dequeue_and_enqueue_token_last1(k)

    def update_teacher_queue_last2(self, teacher, attention_mask):
        t = teacher['hidden_states'][-2][:, 1:, :]
        attention_mask = attention_mask[:, 1:]
        gather_index = torch.nonzero(attention_mask.reshape(-1) == 1).squeeze(1)  # batch x len --
        t = t.reshape(-1, t.shape[-1]).index_select(dim=0, index=gather_index).detach()  # batch x len --, dim
        k = nn.functional.normalize(t, dim=-1)  # batch x len --, dim
        self._dequeue_and_enqueue_token_last2(k)

    def token_contrastive_withmemory(self, s, t, attention_mask, queue):
        # s, t: batch, len, dim
        batch, seqlen, dim = s.shape
        gather_index = torch.nonzero(attention_mask.reshape(-1) == 1).squeeze(1)    # batch x len --
        # select none-pad tokens
        s = s.reshape(-1, dim).index_select(dim=0, index=gather_index)  # batch x len --, dim
        t = t.reshape(-1, dim).index_select(dim=0, index=gather_index).detach()  # batch x len --, dim
        q = nn.functional.normalize(s, dim=-1)   # batch x len --, dim
        k = nn.functional.normalize(t, dim=-1)   # batch x len --, dim
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # cosine similarity for positive
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])  # cosine similarity for negative
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(logits.get_device())
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss


    def full_contrastive(self, student, teacher, attention_mask=None, student_dense=None, t_layerid=None, queue=None):
        student_reps = student['hidden_states']
        teacher_reps = teacher['hidden_states']
        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]
        loss = self.token_contrastive_withmemory(student_dense(student_reps[-1][:, 1:, :]),
                                                                teacher_reps[t_layerid][:, 1:, :],
                                                                attention_mask[:, 1:],
                                                                queue)
        return loss

    def full_attmse(self, student, teacher, attention_mask=None, s_layerids=None, t_layerids=None):
        student_atts = student['attentions']
        teacher_atts = teacher['attentions']
        teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]
        att_loss = 0.
        for s_lid, t_lid in zip(s_layerids, t_layerids):
            student_att = student_atts[s_lid][:, :, 1:, 1:]
            teacher_att = teacher_atts[t_lid][:, :, 1:, 1:]
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student_att),
                                      student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher_att),
                                      teacher_att)
            att_loss += self.loss_mse(student_att, teacher_att)
        loss = att_loss
        return loss

    def align_visible_contrastive(self, student, teacher, attention_mask=None, student_dense=None, t_layerid=None, queue=None):
        student_reps = student['hidden_states']
        teacher_reps = teacher['hidden_states']
        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]
        student_visible = student_reps[-1]
        teacher_visible = expand_gather(teacher_reps[t_layerid], 1, student['all_remain_indices'][-1].unsqueeze(-1))
        visible_mask = expand_gather(attention_mask, 1, student['all_remain_indices'][-1])  # batch, num_remain
        loss_visible = self.token_contrastive_withmemory(student_dense(student_visible)[:, 1:, :],
                                                  teacher_visible[:, 1:, :],
                                                  visible_mask.to(attention_mask)[:, 1:],
                                                  queue)
        return loss_visible

    def align_visible_attmse(self, student, teacher, attention_mask=None, n_relation_heads=12,
                                                        s_layerids=None, t_layerids=None):
        student_qkvs = student['qkvs']
        teacher_qkvs = teacher['qkvs']
        loss = 0.
        for s_lid, t_lid in zip(s_layerids, t_layerids):
            student_qkv = student_qkvs[s_lid]
            teacher_qkv = teacher_qkvs[t_lid]
            num_attention_heads = n_relation_heads
            teacher_head_size = 768 // num_attention_heads
            student_head_size = 384 // num_attention_heads

            student_visible_qkv = student_qkv  # batch, num_remain, dim
            teacher_visible_qkv = [expand_gather(_.detach(), 1, student['all_remain_indices'][s_lid].unsqueeze(-1)) for _ in
                                   teacher_qkv]
            visible_mask = expand_gather(attention_mask, 1, student['all_remain_indices'][s_lid])  # batch, num_remain
            visible_mask = visible_mask.long()  # batch, num_remain, 1
            extended_visible_attention_mask = visible_mask[:, None, None, :]
            extended_visible_attention_mask = (1.0 - extended_visible_attention_mask) * -10000.0

            teacher_query_layer = transpose_for_scores(teacher_visible_qkv[0], num_attention_heads=num_attention_heads,
                                                       attention_head_size=teacher_head_size)
            teacher_key_layer = transpose_for_scores(teacher_visible_qkv[1], num_attention_heads=num_attention_heads,
                                                     attention_head_size=teacher_head_size)

            student_query_layer = transpose_for_scores(student_visible_qkv[0], num_attention_heads=num_attention_heads,
                                                       attention_head_size=student_head_size)
            student_key_layer = transpose_for_scores(student_visible_qkv[1], num_attention_heads=num_attention_heads,
                                                     attention_head_size=student_head_size)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            teacher_attention_scores = torch.matmul(teacher_query_layer, teacher_key_layer.transpose(-1, -2))
            teacher_attention_scores = teacher_attention_scores / math.sqrt(teacher_head_size)
            teacher_attention_scores += extended_visible_attention_mask

            student_attention_scores = torch.matmul(student_query_layer, student_key_layer.transpose(-1, -2))
            student_attention_scores = student_attention_scores / math.sqrt(student_head_size)
            student_attention_scores += extended_visible_attention_mask

            student_attention_scores = student_attention_scores[:, :, 1:, 1:]
            teacher_attention_scores = teacher_attention_scores[:, :, 1:, 1:]
            student_att = torch.where(student_attention_scores <= -1e2,
                                      torch.zeros_like(student_attention_scores).to(student_attention_scores),
                                      student_attention_scores)
            teacher_att = torch.where(teacher_attention_scores <= -1e2,
                                      torch.zeros_like(teacher_attention_scores).to(teacher_attention_scores),
                                      teacher_attention_scores)
            loss += self.loss_mse(student_att, teacher_att)
        return loss

    def reconstruct(self, student, teacher, input_ids=None, attention_mask=None,
                    student_dense_other=None, queue=None):
        teacher_rep = teacher['hidden_states'][-1].detach()
        teacher_encoder_lastlayer = self.teacher.roberta.encoder.layer[-1]

        deleted_indices = torch.cat(student['all_remove_indices'], -1)
        remained_indices = student['all_remain_indices'][-1]

        student_visible = expand_gather(student['last_hidden_state'], 1, remained_indices.unsqueeze(-1))

        position_ids = create_position_ids_from_input_ids(input_ids=input_ids,
                                                          padding_idx=self.student.roberta.embeddings.padding_idx,
                                                          past_key_values_length=0)
        deleted_position_ids = expand_gather(position_ids, 1, deleted_indices)  # batch, num_delete
        masked_input_ids = (torch.ones_like(deleted_position_ids) * self.special_tok_ids["mask_token"]).long()

        student_other = embeddings_forward(self.student.roberta.embeddings, input_ids=masked_input_ids,
                                           position_ids=deleted_position_ids) + \
                        expand_gather(student['last_hidden_state'], 1, deleted_indices.unsqueeze(-1)).detach()

        teacher_visible = expand_gather(teacher_rep, 1, remained_indices.unsqueeze(-1))
        teacher_other = expand_gather(teacher_rep, 1, deleted_indices.unsqueeze(-1))

        visible_mask = expand_gather(attention_mask, 1, remained_indices)  # batch, num_remain
        deleted_mask = expand_gather(attention_mask, 1, deleted_indices)  # batch, num_deleted
        all_mask = torch.cat([visible_mask, deleted_mask], dim=1)  # batch, num_remain + num_deleted
        assert teacher_rep.shape[1] == \
               (visible_mask.shape[1] + deleted_mask.shape[1]) == \
               (student_visible.shape[1] + student_other.shape[1]) == (
                       teacher_visible.shape[1] + teacher_other.shape[1])

        all_mask = all_mask.long()
        extended_final_attention_mask = all_mask[:, None, None, :]
        extended_final_attention_mask = (1.0 - extended_final_attention_mask) * -10000.0
        hidden_input = torch.cat([student_visible, student_other], dim=1)

        layer_outputs, keep_indices, delete_indices = teacher_encoder_lastlayer(
            hidden_states=student_dense_other(hidden_input),
            attention_mask=extended_final_attention_mask,
            output_attentions=True
        )
        hidden_predicted = layer_outputs[0]
        attention_predicted = layer_outputs[1]
        assert attention_predicted.shape[1] == 12

        teacher_gold_rep = torch.cat([teacher_visible, teacher_other], dim=1)
        # contrastive loss
        loss_token_last_withmemory = self.token_contrastive_withmemory(hidden_predicted[:, 1:, :],
                                                                       teacher_gold_rep[:, 1:, :],
                                                                       all_mask[:, 1:],
                                                                       queue=queue)
        # attention loss
        teacher_qkvs = teacher['qkvs'][-1]
        num_attention_heads = 12
        teacher_head_size = 768 // num_attention_heads
        teacher_visible_qkv = [expand_gather(_.detach(), 1, remained_indices.unsqueeze(-1)) for _ in teacher_qkvs]
        teacher_other_qkv = [expand_gather(_.detach(), 1, deleted_indices.unsqueeze(-1)) for _ in teacher_qkvs]

        teacher_q = torch.cat([teacher_visible_qkv[0], teacher_other_qkv[0]], dim=1)
        teacher_k = torch.cat([teacher_visible_qkv[1], teacher_other_qkv[1]], dim=1)
        teacher_query_layer = transpose_for_scores(teacher_q, num_attention_heads=num_attention_heads,
                                                   attention_head_size=teacher_head_size)
        teacher_key_layer = transpose_for_scores(teacher_k, num_attention_heads=num_attention_heads,
                                                 attention_head_size=teacher_head_size)
        teacher_attention_scores = torch.matmul(teacher_query_layer, teacher_key_layer.transpose(-1, -2))
        teacher_attention_scores = teacher_attention_scores / math.sqrt(teacher_head_size)
        teacher_attention_scores += extended_final_attention_mask

        student_att = attention_predicted[:, :, 1:, 1:]
        teacher_att = teacher_attention_scores[:, :, 1:, 1:].detach()
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student_att),
                                  student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher_att),
                                  teacher_att)
        att_loss = self.loss_mse(student_att, teacher_att)

        # overall loss
        loss = loss_token_last_withmemory + att_loss
        if torch.isnan(loss):
            print('NAN')
            loss = 0.
        return loss

    def forward_generaldistill(self, student, teacher, attention_mask=None, input_ids=None, default_batchsize=None):
        # original
        ori_loss_contrastive = self.full_contrastive(student[0], teacher, attention_mask,
                                                     student_dense=self.student.dense_last,
                                                     t_layerid=-2,
                                                     queue=self.queue_token_last2)
        ori_loss_attmse = self.full_attmse(student[0], teacher, attention_mask,
                                           s_layerids=[-1],
                                           t_layerids=[-2])
        ori_loss_all = ori_loss_contrastive + ori_loss_attmse

        # alignment for remaining tokens
        align_loss_contrastive = self.align_visible_contrastive(student[1], teacher, attention_mask,
                                                                student_dense=self.student.dense_last,
                                                                t_layerid=-2,
                                                                queue=self.queue_token_last2)
        align_loss_attmse = self.align_visible_attmse(student[1], teacher, attention_mask,
                                                      s_layerids=[-1],
                                                      t_layerids=[-2])
        align_loss_all = align_loss_contrastive + align_loss_attmse

        reconstruct_loss_all = self.reconstruct(student[1], teacher, input_ids, attention_mask,
                                                student_dense_other=self.student.dense_last_reconstruct,
                                                queue=self.queue_token_last1)

        self.update_teacher_queue_last1(teacher, attention_mask)
        self.update_teacher_queue_last2(teacher, attention_mask)

        loss = ori_loss_all + 0.5 * (align_loss_all + reconstruct_loss_all)
        return loss

    def forward_taskdistill(self, student, teacher):
        student_reps = student[1]['hidden_states']
        teacher_reps = teacher['hidden_states']
        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]

        student_cls = self.student.dense_last(student_reps[-1][:, 0, :])
        teacher_cls = teacher_reps[-1][:, 0, :]

        # student as q, teacher as k
        q = nn.functional.normalize(student_cls, dim=1)
        k = nn.functional.normalize(teacher_cls, dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # cosine similarity for positive
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # cosine similarity for negative

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(logits.get_device())
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue_sent(k)
        return loss

    # general distillation
    def forward(self, student, teacher, attention_mask=None, input_ids=None, default_batchsize=None, training_type='general'):
        if training_type == 'general':
            return self.forward_generaldistill(student, teacher, attention_mask, input_ids, default_batchsize)
        else:
            return self.forward_taskdistill(student, teacher)

    @torch.no_grad()
    def _dequeue_and_enqueue_sent(self, keys):
        # gather keys before updating queue
        if dist.is_initialized():
            keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K_sent:
            keys = keys[:self.K_sent - ptr, :]
            batch_size = keys.shape[0]
            assert ptr + batch_size == self.K_sent
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K_sent  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _gather(self, keys):
        lens = concat_all_gather(torch.tensor(keys.shape[0]).unsqueeze(0).cuda(keys.get_device()))
        max_len = torch.max(lens)
        len_, dim = keys.shape
        if len_ < max_len:
            padding = torch.randn((max_len - len_), dim).cuda(keys.get_device())  # max_len - batch, dim
            new_keys = torch.cat([keys, padding], 0)  # max_len, dim
            keys_mask = torch.cat([torch.ones(len_), torch.zeros(max_len - len_)], 0).long().cuda(keys.get_device())
        else:
            new_keys = keys
            keys_mask = torch.ones(len_).long().cuda(keys.get_device())
        new_keys = concat_all_gather(new_keys)
        keys_mask = concat_all_gather(keys_mask)
        gather_index = torch.nonzero(keys_mask == 1).squeeze(1)
        keys = new_keys.index_select(dim=0, index=gather_index)
        return keys, lens

    @torch.no_grad()
    def _dequeue_and_enqueue_token_last1(self, keys):
        # gather keys before updating queue
        if dist.is_initialized():
            lens = concat_all_gather(torch.tensor(keys.shape[0]).unsqueeze(0).cuda(keys.get_device()))
            max_len = torch.max(lens)
            len_, dim = keys.shape
            if len_ < max_len:
                padding = torch.randn((max_len - len_), dim).cuda(keys.get_device())  # max_len - batch, dim
                new_keys = torch.cat([keys, padding], 0)  # max_len, dim
                keys_mask = torch.cat([torch.ones(len_), torch.zeros(max_len - len_)], 0).long().cuda(keys.get_device())
            else:
                new_keys = keys
                keys_mask = torch.ones(len_).long().cuda(keys.get_device())
            new_keys = concat_all_gather(new_keys)
            keys_mask = concat_all_gather(keys_mask)
            gather_index = torch.nonzero(keys_mask == 1).squeeze(1)
            keys = new_keys.index_select(dim=0, index=gather_index)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_token_last1)

        if ptr + batch_size > self.K:
            keys = keys[:self.K - ptr, :]
            batch_size = keys.shape[0]
            assert ptr + batch_size == self.K
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_token_last1[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_token_last1[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_token_last2(self, keys):
        # gather keys before updating queue
        if dist.is_initialized():
            lens = concat_all_gather(torch.tensor(keys.shape[0]).unsqueeze(0).cuda(keys.get_device()))
            max_len = torch.max(lens)
            len_, dim = keys.shape
            if len_ < max_len:
                padding = torch.randn((max_len - len_), dim).cuda(keys.get_device())  # max_len - batch, dim
                new_keys = torch.cat([keys, padding], 0)  # max_len, dim
                keys_mask = torch.cat([torch.ones(len_), torch.zeros(max_len - len_)], 0).long().cuda(keys.get_device())
            else:
                new_keys = keys
                keys_mask = torch.ones(len_).long().cuda(keys.get_device())
            new_keys = concat_all_gather(new_keys)
            keys_mask = concat_all_gather(keys_mask)
            gather_index = torch.nonzero(keys_mask == 1).squeeze(1)
            keys = new_keys.index_select(dim=0, index=gather_index)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_token_last2)

        if ptr + batch_size > self.K:
            keys = keys[:self.K - ptr, :]
            batch_size = keys.shape[0]
            assert ptr + batch_size == self.K
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_token_last2[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_token_last2[0] = ptr


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq'custom_datasets `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def embeddings_forward(self, input_ids, position_ids):
    input_shape = input_ids.size()
    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    inputs_embeds = self.word_embeddings(input_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids).detach()
    position_embeddings = self.position_embeddings(position_ids).detach()

    assert self.position_embedding_type == "absolute"
    embeddings = inputs_embeds + token_type_embeddings + position_embeddings
    return embeddings

def transpose_for_scores(x, num_attention_heads=12, attention_head_size=64):
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

def expand_gather(input, dim, index):
    size = list(input.size())
    size[dim] = -1
    return input.gather(dim, index.expand(*size))

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
