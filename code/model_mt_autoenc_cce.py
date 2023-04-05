import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from locked_dropout import LockedDropout
from typing import Optional


def sample_gumbel(x):
    noise = torch.cuda.FloatTensor(x.size()).uniform_()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)


def gumbel_softmax_sample(x, tau=0.5):
    noise = sample_gumbel(x)
    y = (F.log_softmax(x, dim=-1) + noise) / tau
    #ysft = F.softmax(y)
    return y.view_as(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TranslatorGeneratorModel(nn.Module):
    def __init__(self,
                 ntoken,
                 ninp,
                 msg_len=64,
                 msg_in_mlp_layers=1,
                 msg_in_mlp_nodes=[],
                 nlayers_encoder=6,
                 transformer_drop=0.1,
                 dropouti=0.15,
                 dropoute=0.1,
                 tie_weights=False,
                 shared_encoder=False,
                 attention_heads=8,
                 pretrained_model=None):
        super(TranslatorGeneratorModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, transformer_drop)

        self.transformer_drop = transformer_drop
        self.embeddings = nn.Embedding(ntoken, ninp)
        self.attention_heads = attention_heads

        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp,
                                                             nhead=attention_heads,
                                                             dropout=transformer_drop)
        self.msg_decoder_layer = nn.TransformerEncoderLayer(d_model=ninp,
                                                            nhead=attention_heads,
                                                            dropout=transformer_drop)
        self.sent_dec_encoder_layer = nn.TransformerDecoderLayer(d_model=ninp,
                                                                 nhead=attention_heads,
                                                                 dropout=transformer_drop)

        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer,
                                                  nlayers_encoder)
        self.msg_decoder = nn.TransformerEncoder(self.msg_decoder_layer, nlayers_encoder)
        self.sent_decoder = nn.TransformerDecoder(self.sent_dec_encoder_layer,
                                                  nlayers_encoder)

        self.tie_weights = tie_weights
        self.msg_in_mlp_layers = msg_in_mlp_layers

        # MLP for the input message
        if msg_in_mlp_layers == 1:
            self.msg_in_mlp = nn.Linear(msg_len, ninp)
        else:
            self.msg_in_mlp = [
                nn.Linear(
                    msg_len if layer == 0 else msg_in_mlp_nodes[layer - 1],
                    msg_in_mlp_nodes[layer] if layer != msg_in_mlp_layers - 1 else ninp)
                for layer in range(msg_in_mlp_layers)
            ]
            self.msg_in_mlp = torch.nn.ModuleList(self.msg_in_mlp)

        # mlp for the message decoding. Takes the last token output
        self.msg_out_mlp = [nn.Linear(2 * ninp, msg_len)]
        self.msg_out_mlp = torch.nn.ModuleList(self.msg_out_mlp)

        if shared_encoder:
            self.msg_decoder = self.sent_encoder

        # decodes the last transformer encoder layer output to vocab.
        self.decoder = nn.Linear(ninp, ntoken)

        if tie_weights:
            self.decoder.weight = self.embeddings.weight
        self.init_weights()
        if pretrained_model:
            self.init_model(pretrained_model)

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_model(self, pretrained_model):
        with torch.no_grad():
            self.embeddings.weight.data = copy.deepcopy(
                pretrained_model.embeddings.weight.data)
            self.sent_encoder = copy.deepcopy(pretrained_model.sent_encoder)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0,
                                        float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_msg_decode(self,
                           input: torch.Tensor,
                           src_key_padding_mask: Optional[torch.Tensor] = None):

        input = input * math.sqrt(self.ninp)
        input = self.pos_encoder(input)
        msg_decoder_out = self.msg_decoder(input,
                                           src_key_padding_mask=src_key_padding_mask)

        m = nn.AdaptiveAvgPool1d(1)

        msg_dec_avg = m(
            msg_decoder_out.view(msg_decoder_out.size(1), msg_decoder_out.size(2),
                                 msg_decoder_out.size(0)))
        msg_dec_avg = msg_dec_avg.view(msg_dec_avg.size(0), msg_dec_avg.size(1))

        last_state = msg_decoder_out[msg_decoder_out.size(0) - 1, :, :]
        msg_decoder_rep = torch.cat([msg_dec_avg, last_state], dim=1)

        for ff in self.msg_out_mlp:
            msg_decoder_rep = ff(msg_decoder_rep)

        decoded_msg_out = msg_decoder_rep

        return decoded_msg_out

    def forward_sent_encoder(self,
                             token_ids: torch.Tensor,
                             msgs: torch.Tensor,
                             gumbel_temp: float,
                             only_embedding: bool = False,
                             src_key_padding_mask: Optional[torch.Tensor] = None):
        if only_embedding:
            emb = self.embeddings(token_ids)
            return emb

        emb = self.embeddings(token_ids) * math.sqrt(self.ninp)
        emb = self.pos_encoder(emb)

        # L, B, H
        sent_encoder_out = self.sent_encoder(emb,
                                             src_key_padding_mask=src_key_padding_mask)
        m = nn.AdaptiveAvgPool1d(1)

        # B, H, 1
        sent_embedding = m(
            sent_encoder_out.view(sent_encoder_out.size(1), sent_encoder_out.size(2),
                                  sent_encoder_out.size(0)))

        # B, H
        sent_embedding = sent_embedding.view(sent_embedding.size(0),
                                             sent_embedding.size(1))

        # get msg fc
        prev_msg_out = msgs

        if self.msg_in_mlp_layers == 1:
            prev_msg_out = F.relu(self.msg_in_mlp(prev_msg_out.float()))
        else:
            for _, ff in enumerate(self.msg_in_mlp):
                prev_msg_out = F.relu(ff(prev_msg_out))
        msg_out = prev_msg_out

        # add the message to the sentence embedding
        both_embeddings = sent_embedding.add(msg_out)
        return both_embeddings, sent_encoder_out

    def forward_sent_decoder(self,
                             both_embeddings: torch.Tensor,
                             token_ids: torch.Tensor,
                             sent_encoder_out: torch.Tensor,
                             gumbel_temp: float,
                             src_key_padding_mask: Optional[torch.Tensor] = None):
        device = token_ids.device
        tgt_mask = self._generate_square_subsequent_mask(len(token_ids)).to(device)

        input_data_emb = self.embeddings(token_ids)

        # L, B, E
        decoder_inputs = torch.zeros_like(input_data_emb)
        decoder_inputs[1:, :, :] = input_data_emb[:-1, :, :]

        if src_key_padding_mask is not None:
            # B, L
            tgt_key_padding_mask = torch.zeros_like(src_key_padding_mask).bool()
            tgt_key_padding_mask[:, 1:] = src_key_padding_mask[:, :-1]
        else:
            tgt_key_padding_mask = None

        both_embeddings_repeat = both_embeddings.view(1, both_embeddings.size(0),
                                                      both_embeddings.size(1))
        both_embeddings_repeat = both_embeddings_repeat.repeat(token_ids.size(0), 1, 1)
        decoder_inputs = decoder_inputs + both_embeddings_repeat
        decoder_inputs = self.pos_encoder(decoder_inputs)

        # sent_decoded: L, B, H
        sent_decoded = self.sent_decoder(decoder_inputs,
                                         memory=sent_encoder_out,
                                         tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=tgt_key_padding_mask,
                                         memory_key_padding_mask=src_key_padding_mask)
        # sent_decoded_vocab: LB, V
        sent_decoded_vocab = self.decoder(
            sent_decoded.view(
                sent_decoded.size(0) * sent_decoded.size(1), sent_decoded.size(2)))
        # sent_decoded_vocab_hot: LB, V
        sent_decoded_vocab_hot = F.gumbel_softmax(F.log_softmax(sent_decoded_vocab,
                                                                dim=-1),
                                                  tau=gumbel_temp,
                                                  hard=True)
        # sent_decoded_vocab_hot_out: L, B, V
        sent_decoded_vocab_hot_out = sent_decoded_vocab_hot.view(
            decoder_inputs.size(0), decoder_inputs.size(1),
            sent_decoded_vocab_hot.size(1))

        sent_decoded_vocab_emb = torch.mm(sent_decoded_vocab_hot, self.embeddings.weight)
        # sent_decoded_vocab_emb: L, B, H
        sent_decoded_vocab_emb = sent_decoded_vocab_emb.view(decoder_inputs.size(0),
                                                             decoder_inputs.size(1),
                                                             decoder_inputs.size(2))

        sent_decoded_vocab_soft = gumbel_softmax_sample(sent_decoded_vocab,
                                                        tau=gumbel_temp)

        return sent_decoded_vocab_emb, sent_decoded_vocab_hot_out, sent_decoded_vocab_soft

    def forward_sent(self,
                     token_ids: torch.Tensor,
                     msgs: torch.Tensor,
                     gumbel_temp: float,
                     only_embedding: bool = False,
                     src_key_padding_mask: Optional[torch.Tensor] = None):
        if only_embedding:
            return self.forward_sent_encoder(token_ids,
                                             msgs,
                                             gumbel_temp,
                                             only_embedding=True,
                                             src_key_padding_mask=src_key_padding_mask)

        sent_msg_embedding, encoder_out = self.forward_sent_encoder(
            token_ids,
            msgs,
            gumbel_temp,
            only_embedding=False,
            src_key_padding_mask=src_key_padding_mask)
        return self.forward_sent_decoder(sent_msg_embedding,
                                         token_ids,
                                         encoder_out,
                                         gumbel_temp,
                                         src_key_padding_mask=src_key_padding_mask)


class TranslatorDiscriminatorModel(nn.Module):
    def __init__(self,
                 ninp,
                 nlayers_encoder=6,
                 transformer_drop=0.1,
                 attention_heads=8,
                 dropouti=0.1):
        # directly takes the embedding
        super(TranslatorDiscriminatorModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.ninp = ninp
        self.transformer_drop = transformer_drop
        self.attention_heads = attention_heads
        self.pos_encoder = PositionalEncoding(ninp, transformer_drop)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=ninp,
                                                        nhead=attention_heads,
                                                        dropout=transformer_drop)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         nlayers_encoder)

        # classification to fake and real.
        self.real_fake_classify = [nn.Linear(ninp * 2, 1)]
        self.real_fake_classify = torch.nn.ModuleList(self.real_fake_classify)

        self.init_weights()

    def init_weights(self):
        for ff in self.real_fake_classify:
            torch.nn.init.xavier_normal_(ff.weight)
            ff.bias.data.fill_(0.01)

    def forward(self,
                input_emb: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        input_emb = input_emb * math.sqrt(self.ninp)
        input_emb = self.pos_encoder(input_emb)
        discr_encoder_out = self.transformer_encoder(
            input_emb, src_key_padding_mask=src_key_padding_mask)

        last_state = discr_encoder_out[discr_encoder_out.size(0) - 1, :, :]

        m = nn.AdaptiveAvgPool1d(1)
        disc_avg = m(
            discr_encoder_out.view(discr_encoder_out.size(1), discr_encoder_out.size(2),
                                   discr_encoder_out.size(0)))
        disc_avg = disc_avg.view(disc_avg.size(0), disc_avg.size(1))

        disc_enc_rep = torch.cat([disc_avg, last_state], dim=1)
        for ff in self.real_fake_classify:
            disc_enc_rep = ff(disc_enc_rep)

        real_fake_out = disc_enc_rep
        return real_fake_out
