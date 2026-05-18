import torch
import torch.nn as nn

from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from models.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        heads,
        encoder_layers,
        decoder_layers,
        d_ff,
        dropout,
        pad_idx
    ):
        super().__init__()

        self.src_embedding=nn.Embedding(src_vocab_size,d_model)
        self.tgt_embedding=nn.Embedding(tgt_vocab_size,d_model)

        self.positional_encoding=PositionalEncoding(d_model)

        self.encoder_layers=nn.ModuleList([
            EncoderLayer(d_model,heads,d_ff,dropout)
            for _ in range(encoder_layers)
        ])

        self.decoder_layers=nn.ModuleList([
            DecoderLayer(d_model,heads,d_ff,dropout)
            for _ in range(decoder_layers)
        ])

        self.fc_out=nn.Linear(d_model,tgt_vocab_size)

        self.dropout=nn.Dropout(dropout)

        self.pad_idx=pad_idx

    def create_padding_mask(self,seq):
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self,size):
        return torch.tril(torch.ones(size,size)).bool()

    def forward(self,src,tgt):
        src_mask=self.create_padding_mask(src)

        tgt_padding_mask=self.create_padding_mask(tgt)

        look_ahead_mask=self.create_look_ahead_mask(
            tgt.size(1)
        ).to(tgt.device)

        tgt_mask=tgt_padding_mask & look_ahead_mask

        src=self.dropout(
            self.positional_encoding(
                self.src_embedding(src)
            )
        )

        tgt=self.dropout(
            self.positional_encoding(
                self.tgt_embedding(tgt)
            )
        )

        enc_output=src

        for layer in self.encoder_layers:
            enc_output=layer(enc_output,src_mask)

        dec_output=tgt

        attention=None

        for layer in self.decoder_layers:
            dec_output,attention=layer(
                dec_output,
                enc_output,
                src_mask,
                tgt_mask
            )

        output=self.fc_out(dec_output)

        return output,attention
