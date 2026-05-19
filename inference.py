import torch

from models.transformer import Transformer

from utils import (
    src_vocab,
    tgt_vocab
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=256,
    heads=8,
    encoder_layers=4,
    decoder_layers=4,
    d_ff=1024,
    dropout=0.1,
    pad_idx=0
).to(device)

model.load_state_dict(
    torch.load(
        "transformer.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")