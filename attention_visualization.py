import torch
import matplotlib.pyplot as plt
import seaborn as sns

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

sentence = "ein kleines mädchen spielt im park"

tokens = tokenize_de(sentence)

src = text_to_tensor(
    tokens,
    src_vocab
).unsqueeze(0).to(device)

src_mask = model.make_src_mask(src)

with torch.no_grad():

    encoder_output = model.encoder(
        src,
        src_mask
    )

last_layer = model.encoder.layers[-1]

attention_weights = last_layer.attention_weights.squeeze(0)

tokens = ["<sos>"] + tokens + ["<eos>"]

for head in range(attention_weights.shape[0]):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        attention_weights[head].cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis"
    )

    plt.title(f"Attention Head {head}")

    plt.xlabel("Key Tokens")

    plt.ylabel("Query Tokens")

    plt.tight_layout()

    plt.savefig(
        f"head_{head}.png"
    )

    plt.close()

print("Attention heatmaps saved successfully")