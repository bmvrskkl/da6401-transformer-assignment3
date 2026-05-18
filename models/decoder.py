import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.positional_encoding import PositionalEncoding

class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        heads,
        layers,
        d_ff,
        dropout,
        max_len=5000
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position = PositionalEncoding(
            d_model,
            max_len
        )

        self.layers = nn.ModuleList([
            MultiHeadAttention(
                d_model,
                heads
            )
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        enc_out,
        src_mask,
        tgt_mask
    ):

        x = self.embedding(x)

        x = self.position(x)

        x = self.dropout(x)

        for layer in self.layers:

            attention = layer(
                x,
                x,
                x,
                tgt_mask
            )

            x = self.norm(
                x + attention
            )

        return x