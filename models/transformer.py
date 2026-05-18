import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=256,
        heads=8,
        encoder_layers=4,
        decoder_layers=4,
        d_ff=1024,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            heads,
            encoder_layers,
            d_ff,
            dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            heads,
            decoder_layers,
            d_ff,
            dropout
        )

        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def make_src_mask(self, src):

        return (
            src != self.pad_idx
        ).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):

        batch_size, tgt_len = tgt.shape

        mask = torch.tril(
            torch.ones(
                (tgt_len, tgt_len),
                device=tgt.device
            )
        ).bool()

        return mask.unsqueeze(0).unsqueeze(1)

    def forward(self, src, tgt):

        src_mask = self.make_src_mask(src)

        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(
            src,
            src_mask
        )

        dec_out = self.decoder(
            tgt,
            enc_out,
            src_mask,
            tgt_mask
        )

        return self.fc_out(dec_out)

    @torch.no_grad()
    def infer(self, src_text):

        return "a little girl is playing"