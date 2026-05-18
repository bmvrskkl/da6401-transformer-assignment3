import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
from tqdm import tqdm
from sacrebleu import corpus_bleu

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

wandb.init(project="da6401_transformer")

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

class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.1):
        super().__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):

        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():

            true_dist = torch.zeros_like(pred)

            true_dist.fill_(
                self.smoothing / (self.cls - 1)
            )

            true_dist.scatter_(
                1,
                target.data.unsqueeze(1),
                self.confidence
            )

        return torch.mean(
            torch.sum(-true_dist * pred, dim=-1)
        )

model = Transformer(
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

optimizer = torch.optim.Adam(
    model.parameters(),
    betas=(0.9, 0.98),
    eps=1e-9
)

scheduler = NoamScheduler(
    optimizer,
    D_MODEL,
    WARMUP_STEPS
)

criterion = LabelSmoothingLoss(
    len(tgt_vocab),
    LABEL_SMOOTHING
)

best_bleu = 0

@torch.no_grad()
def evaluate():

    model.eval()

    predictions = []
    references = []

    total_val_loss = 0

    for batch in valid_loader:

        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output, _ = model(src, tgt_input)

        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt_output.reshape(-1)
        )

        total_val_loss += loss.item()

        predicted_ids = output.argmax(-1)

        for pred, ref in zip(predicted_ids, tgt_output):

            pred_tokens = []
            ref_tokens = []

            for idx in pred:

                token = tgt_vocab.lookup_token(idx.item())

                if token == "<eos>":
                    break

                if token not in ["<bos>", "<pad>"]:
                    pred_tokens.append(token)

            for idx in ref:

                token = tgt_vocab.lookup_token(idx.item())

                if token == "<eos>":
                    break

                if token not in ["<bos>", "<pad>"]:
                    ref_tokens.append(token)

            predictions.append(" ".join(pred_tokens))
            references.append([" ".join(ref_tokens)])

    bleu = corpus_bleu(
        predictions,
        references
    ).score

    avg_val_loss = total_val_loss / len(valid_loader)

    return avg_val_loss, bleu

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    loop = tqdm(train_loader)

    for batch in loop:

        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output, _ = model(src, tgt_input)

        output = output.reshape(
            -1,
            output.shape[-1]
        )

        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)

        scheduler.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scheduler.step()

        total_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch+1}"
        )

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)

    val_loss, bleu = evaluate()

    wandb.log({
        "train_loss": avg_loss,
        "val_loss": val_loss,
        "bleu": bleu
    })

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"BLEU: {bleu:.2f}"
    )

    if bleu > best_bleu:

        best_bleu = bleu

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

print("Training complete")