import torch

from config import *

from models.transformer import Transformer

from utils import (
    src_vocab,
    tgt_vocab,
    tokenize_de
)

model=Transformer(
    len(src_vocab),
    len(tgt_vocab),
    D_MODEL,
    NUM_HEADS,
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    D_FF,
    DROPOUT,
    PAD_IDX
).to(DEVICE)

model.load_state_dict(
    torch.load(CHECKPOINT_PATH)
)

model.eval()

def encode(sentence):
    tokens=tokenize_de(sentence)

    ids=[BOS_IDX]

    ids += [src_vocab[token] for token in tokens]

    ids.append(EOS_IDX)

    return torch.tensor(ids).unsqueeze(0)

@torch.no_grad()
def greedy_decode(src,max_len=30):
    generated=torch.tensor([[BOS_IDX]]).to(DEVICE)

    src=src.to(DEVICE)

    for _ in range(max_len):
        output,_=model(src,generated)

        next_token=output[:,-1].argmax(-1).unsqueeze(1)

        generated=torch.cat(
            [generated,next_token],
            dim=1
        )

        if next_token.item()==EOS_IDX:
            break

    return generated

sentence="ein hund läuft im park"

src=encode(sentence)

output=greedy_decode(src)

tokens=[]

for idx in output[0]:
    word=tgt_vocab.lookup_token(idx.item())
    tokens.append(word)

print(" ".join(tokens))
