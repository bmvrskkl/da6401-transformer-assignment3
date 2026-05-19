import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import wandb

from config import *

from models.transformer import Transformer
from models.scheduler import NoamScheduler

from utils import (
    train_data,
    valid_data,
    collate_fn,
    src_vocab,
    tgt_vocab
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

wandb.init(
    project="da6401_transformer"
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
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

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.98),
    eps=1e-9
)

scheduler = NoamScheduler(
    optimizer,
    256,
    4000
)

def decode_tokens(tokens):

    words = []

    for token in tokens:

        token = token.item()

        if token == 2:
            break

        if token > 3:

            words.append(
                tgt_vocab.lookup_token(token)
            )

    return words

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0

    references = []
    hypotheses = []

    for batch in valid_loader:

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(
            src,
            tgt_input
        )

        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt_output.reshape(-1)
        )

        total_loss += loss.item()

        predictions = output.argmax(dim=-1)

        for pred, target in zip(
            predictions,
            tgt_output
        ):

            pred_words = decode_tokens(pred)

            target_words = decode_tokens(target)

            hypotheses.append(pred_words)

            references.append([target_words])

    bleu = corpus_bleu(
        references,
        hypotheses
    ) * 100

    return (
        total_loss / len(valid_loader),
        bleu
    )

EPOCHS = 15

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}"
    )

    for batch in progress_bar:

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        output = model(
            src,
            tgt_input
        )

        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt_output.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    val_loss, bleu = evaluate()

    avg_train_loss = (
        total_loss / len(train_loader)
    )

    wandb.log({
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "bleu": bleu
    })

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"BLEU: {bleu:.2f}"
    )

torch.save(
    model.state_dict(),
    "transformer.pth"
)

print("Training complete")