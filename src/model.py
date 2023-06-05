#!/usr/bin/env python
# -*- coding: utf-8 -*-


from itertools import accumulate

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.Roberta import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SubLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SubLayer, self).__init__()
        head_size = int(d_model / num_heads)
        self.attention = MultiHeadAttention(
            num_heads, d_model, head_size, head_size, dropout
        )
        self.layer_norm_pre = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layer_norm_post = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.attention(x, x, x, mask)[0] + x
        x = self.layer_norm_post(self.ffn(x) + x)
        x = self.dropout(x)
        return x


class SubLayer2(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SubLayer2, self).__init__()
        head_size = int(d_model / num_heads)
        self.attention = MultiHeadAttention(
            num_heads, d_model, head_size, head_size, dropout
        )

    def forward(self, x, mask):
        x = self.attention(x, x, x, mask)[0] + x
        return x


class BertWordPair(nn.Module):
    def __init__(self, config):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        bert_config = AutoConfig.from_pretrained(config.bert_path)

        self.inner_dim = 256

        self.dense0 = nn.Linear(bert_config.hidden_size, self.inner_dim * 4 * 6)
        self.dense1 = nn.Linear(bert_config.hidden_size, self.inner_dim * 4 * 3)
        self.dense2 = nn.Linear(bert_config.hidden_size, self.inner_dim * 4 * 4)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self.sublayer1 = SubLayer(
            bert_config.hidden_size,
            bert_config.num_attention_heads,
            bert_config.hidden_dropout_prob,
        )
        self.sublayer2 = SubLayer(
            bert_config.hidden_size,
            bert_config.num_attention_heads,
            bert_config.hidden_dropout_prob,
        )
        self.sublayer3 = SubLayer2(
            bert_config.hidden_size,
            bert_config.num_attention_heads,
            bert_config.hidden_dropout_prob,
        )

        self.config = config

    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        """
        output_dim = self.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(
            self.config.device
        )
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((1, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))
        embeddings = embeddings.squeeze(0)
        return embeddings

    def get_instance_embedding(
        self,
        qw: torch.Tensor,
        kw: torch.Tensor,
        token_index,
        thread_length,
        pos_type,
    ):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        """

        seq_len, num_classes = qw.shape[:2]

        accu_index = [0] + list(accumulate(thread_length))

        logits = qw.new_zeros([seq_len, seq_len, num_classes])

        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i + 1]
                cstart, cend = accu_index[j], accu_index[j + 1]

                cur_qw, cur_kw = qw[rstart:rend], kw[cstart:cend]
                x, y = token_index[rstart:rend], token_index[cstart:cend]

                # This is used to compute relative distance, see the matrix \
                # in Fig.8 of our paper
                x = -x if i > 0 and i < j else x
                y = -y if j > 0 and i > j else y

                x_pos_emb = self.custom_sinusoidal_position_embedding(x, pos_type)
                y_pos_emb = self.custom_sinusoidal_position_embedding(y, pos_type)

                # Refer to https://kexue.fm/archives/8265
                x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                x_sin_pos = x_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_qw2 = torch.stack([-cur_qw[..., 1::2], cur_qw[..., ::2]], -1)
                cur_qw2 = cur_qw2.reshape(cur_qw.shape)
                cur_qw = cur_qw * x_cos_pos + cur_qw2 * x_sin_pos

                y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                y_sin_pos = y_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_kw2 = torch.stack([-cur_kw[..., 1::2], cur_kw[..., ::2]], -1)
                cur_kw2 = cur_kw2.reshape(cur_kw.shape)
                cur_kw = cur_kw * y_cos_pos + cur_kw2 * y_sin_pos

                pred_logits = torch.einsum("mhd,nhd->mnh", cur_qw, cur_kw).contiguous()
                logits[rstart:rend, cstart:cend] = pred_logits

        return logits

    def get_ro_embedding(self, qw, kw, token_index, thread_lengths, pos_type):
        # qw_res = qw.new_zeros(*qw.shape)
        # kw_res = kw.new_zeros(*kw.shape)
        logits = []
        batch_size = qw.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding(
                qw[i], kw[i], token_index[i], thread_lengths[i], pos_type
            )
            logits.append(pred_logits)
        logits = torch.stack(logits)
        return logits

    def classify_matrix(
        self, kwargs, sequence_outputs, input_labels, masks, mat_name="ent"
    ):
        utterance_index, token_index, thread_lengths = [
            kwargs[w] for w in ["utterance_index", "token_index", "thread_lengths"]
        ]
        if mat_name == "ent":
            outputs = self.dense0(sequence_outputs)
        elif mat_name == "rel":
            outputs = self.dense1(sequence_outputs)
        else:
            outputs = self.dense2(sequence_outputs)

        outputs = torch.split(outputs, self.inner_dim * 4, dim=-1)

        outputs = torch.stack(outputs, dim=-2)

        q_token, q_utterance, k_token, k_utterance = torch.split(
            outputs, self.inner_dim, dim=-1
        )

        if self.config.use_rope:
            if mat_name == "ent":
                pred_logits = self.get_ro_embedding(
                    q_token, k_token, token_index, thread_lengths, pos_type=0
                )  # pos_type=0 for token-level relative distance encoding
            else:
                pred_logits0 = self.get_ro_embedding(
                    q_token, k_token, token_index, thread_lengths, pos_type=0
                )
                pred_logits1 = self.get_ro_embedding(
                    q_utterance,
                    k_utterance,
                    utterance_index,
                    thread_lengths,
                    pos_type=1,
                )  # pos_type=1 for utterance-level relative distance encoding
                pred_logits = pred_logits0 + pred_logits1
        else:
            # without rope, use dot-product attention directly
            pred_logits = torch.einsum("bmhd,bnhd->bmnh", q_token, k_token).contiguous()

        nums = pred_logits.shape[-1]

        criterion = nn.CrossEntropyLoss(
            sequence_outputs.new_tensor(
                [1.0] + [self.config.loss_weight[mat_name]] * (nums - 1)
            )
        )

        criterion = nn.CrossEntropyLoss()

        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)

        return loss, pred_logits

    def build_attention(
        self,
        sequence_outputs,
        speaker_masks=None,
        reply_masks=None,
        thread_masks=None,
    ):
        """
        sequence_outputs: batch_size, seq_len, hidden_size
        speaker_matrix: batch_size, num, num
        head_matrix: batch_size, num, num
        """
        speaker_masks = speaker_masks.bool().unsqueeze(1)
        reply_masks = reply_masks.bool().unsqueeze(1)
        thread_masks = thread_masks.bool().unsqueeze(1)
        # import pdb; pdb.set_trace()
        sequence_outputs = self.sublayer1(sequence_outputs, speaker_masks)
        sequence_outputs = self.sublayer2(sequence_outputs, reply_masks)
        sequence_outputs = self.sublayer3(sequence_outputs, thread_masks)

        return sequence_outputs

    def merge_sentence(self, sequence_outputs, input_masks, dialogue_length):
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                stack.append(sequence_outputs[j, :lens])
            res.append(torch.cat(stack))
        new_res = sequence_outputs.new_zeros(
            [len(res), max(map(len, res)), sequence_outputs.shape[-1]]
        )
        for i, w in enumerate(res):
            new_res[i, : len(w)] = w
        return new_res

    def forward(self, **kwargs):
        input_ids, input_masks, input_segments = [
            kwargs[w] for w in ["input_ids", "input_masks", "input_segments"]
        ]
        ent_matrix, rel_matrix, pol_matrix = [
            kwargs[w] for w in ["ent_matrix", "rel_matrix", "pol_matrix"]
        ]
        reply_masks, speaker_masks, thread_masks = [
            kwargs[w] for w in ["reply_masks", "speaker_masks", "thread_masks"]
        ]
        sentence_masks, full_masks, dialogue_length = [
            kwargs[w] for w in ["sentence_masks", "full_masks", "dialogue_length"]
        ]

        sequence_outputs = self.bert(
            input_ids,
            token_type_ids=input_segments,
            attention_mask=input_masks,
        )[0]

        sequence_outputs = self.merge_sentence(
            sequence_outputs, input_masks, dialogue_length
        )
        sequence_outputs = self.dropout(sequence_outputs)

        sequence_outputs = self.build_attention(
            sequence_outputs,
            reply_masks=reply_masks,
            speaker_masks=speaker_masks,
            thread_masks=thread_masks,
        )

        loss0, tags0 = self.classify_matrix(
            kwargs, sequence_outputs, ent_matrix, sentence_masks, "ent"
        )
        loss1, tags1 = self.classify_matrix(
            kwargs, sequence_outputs, rel_matrix, full_masks, "rel"
        )
        loss2, tags2 = self.classify_matrix(
            kwargs, sequence_outputs, pol_matrix, full_masks, "pol"
        )

        return (loss0, loss1, loss2), (tags0, tags1, tags2)
