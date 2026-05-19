import torch

from models.transformer import Transformer

from utils import (
    tokenize_de,
    src_vocab,
    tgt_vocab,
    text_to_tensor
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

sentence = "ein kleines mädchen spielt"

tokens = tokenize_de(sentence)

src = text_to_tensor(
    tokens,
    src_vocab
).unsqueeze(0).to(device)

tgt = torch.tensor([[1]]).to(device)

generated_tokens = []

with torch.no_grad():

    for _ in range(20):

        output = model(src, tgt)

        next_token = output[:, -1, :].argmax(-1)

        token_id = next_token.item()

        if token_id == 2:
            break

        generated_tokens.append(
            tgt_vocab.lookup_token(token_id)
        )

        tgt = torch.cat(
            [tgt, next_token.unsqueeze(0)],
            dim=1
        )

print("\nGerman Input:")
print(sentence)

print("\nPredicted Translation:")
print(" ".join(generated_tokens))