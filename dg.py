# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None, highlights=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if highlights is not None:
            assert p_attn.size(0) == highlights.size(0)
            assert p_attn.size(-1) == highlights.size(-1)
            p_attn = p_attn * highlights
            if mask is not None:
                p_attn = p_attn.masked_fill(mask == 0, 1e-7)
            p_sum = p_attn.sum(dim=-1, keepdim=True)
            p_attn = p_attn / p_sum

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn



@register_model('dg')
class DGModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on
        parser.add_argument('--proj_init_state', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            weights = np.load(embed_path)
            weights = torch.tensor(weights).float()
            assert list(weights.size()) == [num_embeddings, embed_dim]
            return nn.Embedding.from_pretrained(weights, freeze=False)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = DGEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = DGDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            proj_initial_state=args.proj_init_state
        )
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, q_tokens, q_lengths, ans_tokens, ans_lengths, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, q_tokens=q_tokens, q_lengths=q_lengths,
                                   ans_tokens=ans_tokens, ans_lengths=ans_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def create_mask(x1_len, x2_len, x1_max_len, x2_max_len):
    """
    :param x1_len: [bsz, ]
    :param x2_len: [bsz, ]
    :return: shape: [bsz, x1_max_len, x2_max_len]
    """
    bsz = x1_len.size(0)
    x1_mask = length_to_mask(x1_len, x1_max_len)
    x2_mask = length_to_mask(x2_len, x2_max_len)

    # shape: [bsz, x1_max_len, 1]
    x1_mask = x1_mask.view(bsz, 1, -1).transpose(-2, -1)
    # shape: [bsz, 1, x2_max_len]
    x2_mask = x2_mask.view(bsz, -1).view(bsz, 1, -1)

    return x1_mask * x2_mask


class DGEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

        self.attention = Attention()
        self.dropout = nn.Dropout(dropout_in)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.output_units*2, self.output_units),
            nn.Tanh()
        )
        self.distance_layer = nn.Bilinear(self.output_units, self.output_units, 1)
        self.gating_layer = nn.Linear(self.output_units, 1)
        self.relu = nn.ReLU()

    def lstm_encode(self, tokens, lengths, need_sort=False, need_embed=True):
        """
        :param tokens: shape [bsz, seq_len]
        :param lengths: shape [bsz, ]
        :param need_sort: bool, src tokens已经排好顺序了
        :param need_embed: bool, fusion进来不需要再embed
        :return: x -> shape [bsz, seq_len, *]
                 final_hiddens -> shape [layer, bsz, *]
                 final_cells -> shape [layer, bsz, *]
        """
        bsz, seqlen = tokens.size(0), tokens.size(1)

        if need_embed:
            x = self.embed_tokens(tokens)
            x = F.dropout(x, p=self.dropout_in, training=self.training)
        else:
            x = tokens

        if need_sort:
            x_len_sorted, x_idx = torch.sort(lengths, descending=True)
            x_sorted = x.index_select(dim=0, index=x_idx)
            _, x_ori_idx = torch.sort(x_idx)
        else:
            x_sorted, x_len_sorted = x, lengths

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted.cpu(), batch_first=True)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)

        packed_outs, (final_hiddens, final_cells) = self.lstm(x_packed, (h0, c0))

        x = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value, batch_first=True)[0]
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        if need_sort:
            x = x.index_select(dim=0, index=x_ori_idx)
            final_hiddens = final_hiddens.index_select(dim=1, index=x_ori_idx)
            final_cells = final_cells.index_select(dim=1, index=x_ori_idx)
        assert list(x.size()) == [bsz, seqlen, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        return x, final_hiddens, final_cells

    def forward(self, src_tokens, src_lengths, q_tokens, q_lengths, ans_tokens, ans_lengths):
        # pad_emb = self.embed_tokens(torch.LongTensor([self.padding_idx, ]).cuda())
        # print(pad_emb[0][:10])
        bsz, seqlen = src_tokens.size()
        _, q_seqlen = q_tokens.size()
        _, ans_seqlen = ans_tokens.size()

        src_x, _, _ = self.lstm_encode(src_tokens, src_lengths)
        q_x, q_hiddens, q_cells = self.lstm_encode(q_tokens, q_lengths, need_sort=True)
        ans_x, ans_hiddens, ans_cells = self.lstm_encode(ans_tokens, ans_lengths, need_sort=True)

        # doc attend到Q和A
        mask_for_attend_doc_to_a = create_mask(ans_lengths, src_lengths, ans_seqlen, seqlen)
        a_attend, _ = self.attention(query=ans_x,
                                     key=src_x,
                                     value=src_x,
                                     mask=mask_for_attend_doc_to_a,
                                     dropout=self.dropout)
        ans_x = self.fusion_layer(torch.cat([ans_x, a_attend], dim=-1))

        mask_for_attend_doc_to_q = create_mask(q_lengths, src_lengths, q_seqlen, seqlen)
        q_attend, _ = self.attention(query=q_x,
                                     key=src_x,
                                     value=src_x,
                                     mask=mask_for_attend_doc_to_q,
                                     dropout=self.dropout)
        q_x = self.fusion_layer(torch.cat([q_x, q_attend], dim=-1))

        ans_mask_for_self_attend = length_to_mask(ans_lengths, ans_seqlen)
        # shape: [bsz, hidden*2]
        ans_self_attend = self._self_attend(ans_x, ans_mask_for_self_attend)
        ans_self_attend = ans_self_attend.unsqueeze(1)
        # shape: [bsz, q_seqlen, 1]
        q_ans_distance = self.distance_layer(q_x, ans_self_attend.repeat(1, q_seqlen, 1))
        # highlight出和answer不想关
        q_scored = q_x * q_ans_distance

        # 为decode做初始化
        # shape: [layers, bsz, hidden*2]

        mask_for_attend_q_to_a = create_mask(ans_lengths, q_lengths, ans_seqlen, q_seqlen)
        a_attend, _ = self.attention(query=ans_x,
                                     key=q_x,
                                     value=q_x,
                                     mask=mask_for_attend_q_to_a,
                                     dropout=self.dropout)
        a_fusion = self.fusion_layer(torch.cat([ans_x, a_attend], dim=-1))
        ans_fusion_self_attend = self._self_attend(a_fusion, ans_mask_for_self_attend)
        ans_fusion_self_attend = ans_fusion_self_attend.unsqueeze(1)

        # 找出和answer无关的doc的句子，用来做最后的生成
        # print(src_x.size(), ans_fusion_self_attend.size())
        doc_ans_distance = self.distance_layer(src_x.contiguous(), ans_fusion_self_attend.repeat(1, seqlen, 1))
        doc_scored = src_x * doc_ans_distance

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        mask_for_attend_q_to_doc = create_mask(src_lengths, q_lengths, seqlen, q_seqlen)
        src_attend, _ = self.attention(query=doc_scored,
                                       key=q_scored,
                                       value=q_scored,
                                       mask=mask_for_attend_q_to_doc,
                                       dropout=self.dropout)
        x_fusion = self.fusion_layer(torch.cat([doc_scored, src_attend],
                                               dim=-1))

        # q_doc_attend, _ = self.attention(query=q_scored,
        #                                key=doc_scored,
        #                                value=doc_scored,
        #                                mask=mask_for_attend_q_to_doc.transpose(-1, -2),
        #                                dropout=self.dropout)
        # q_scored = self.fusion_layer(torch.cat([q_scored, q_doc_attend],
        #                                        dim=-1))
        q_scored_outs, q_scored_hiddens, q_scored_cells = self.lstm_encode(q_scored, q_lengths, need_sort=True, need_embed=False)
        
        attend_x, attend_hiddens, attend_cells = self.lstm_encode(x_fusion, src_lengths, need_embed=False)
        attend_x = attend_x.transpose(0, 1)

        q_fusion = q_scored_outs.transpose(0, 1)
        q_mask_for_self_attend = length_to_mask(q_lengths, q_seqlen)
        q_mask_for_self_attend = q_mask_for_self_attend.transpose(0, 1)

        return {
            'encoder_out': (attend_x, attend_hiddens, attend_cells, q_scored_hiddens, q_scored_cells, q_fusion, q_mask_for_self_attend),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def _self_attend(self, seqs, mask=None):
        scores = self.gating_layer(seqs)
        scores = scores.squeeze()

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = attn.unsqueeze(2)

        seqs_scored = seqs * attn
        return torch.sum(seqs_scored, dim=1)

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x1 = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x1.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x1 = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        # x2 = self.input_proj2(input)
        # # compute attention
        # attn_scores2 = (source2 * x2.unsqueeze(0)).sum(dim=2)
        #
        # # don't attend over padding
        # if mask2 is not None:
        #     attn_scores2 = attn_scores2.float().masked_fill_(
        #         mask2,
        #         float('-inf')
        #     ).type_as(attn_scores2)  # FP16 support: cast to float and back
        #
        # attn_scores2 = F.softmax(attn_scores2, dim=0)  # srclen x bsz
        #
        # # sum weighted sources
        # x2 = (attn_scores2.unsqueeze(2) * source2).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x1, input), dim=1)))
        return x, attn_scores


class DGDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None, proj_initial_state=False
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        # if hidden_size != out_embed_dim:
        # self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)
        self.match_layer = nn.Bilinear(hidden_size, hidden_size, 1)
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

        self.proj_initial_state = proj_initial_state
        if proj_initial_state:
            self.proj_hiddens = Linear(encoder_output_units, encoder_output_units)
            self.proj_cells = Linear(encoder_output_units, encoder_output_units)

    def _self_attend(self, seqs, query, mask=None):
        # return [bsz, hidden_size], [bsz, seqlen]
        bsz, seqlen, d1 = seqs.size()
        scores = self.match_layer(seqs, query.unsqueeze(1).repeat(1, seqlen, 1))
        scores = self.relu(scores)
        scores = scores.squeeze()

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = attn.unsqueeze(2)

        seqs_scored = seqs * attn
        return torch.sum(seqs_scored, dim=1), attn.squeeze()

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        first_hiddens, first_cells = encoder_out[3:5]
        q_fusion, q_mask = encoder_out[5:]

        q_fusion = q_fusion.transpose(0, 1).contiguous()
        q_mask = q_mask.transpose(0, 1)
        srclen = encoder_outs.size(0)

        if self.proj_initial_state:
            first_hiddens = self.proj_hiddens(first_hiddens)
            first_cells = self.proj_cells(first_cells)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [first_hiddens[0] for i in range(num_layers)]
            prev_cells = [first_cells[0] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                # q_self_attend, _ = self._self_attend(q_fusion, hidden, mask=q_mask)
                # gates = self.gate_layer(hidden)
                # out2, _ = self.attention(q_self_attend, encoder_outs, encoder_padding_mask)
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
                # out = gates * out + (1-gates) * out2
                # out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x + 1e-7, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('dg', 'dg')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')
