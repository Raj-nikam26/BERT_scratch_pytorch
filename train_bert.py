import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import os
from model_parts import EncoderStack, LayerNorm
from config_bert import BERT_config
from BERT import BERTModel

os.makedirs("artifacts/bert_base", exist_ok=True)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET (same as yours, unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BERTPretrainDataset(Dataset):
    def __init__(self, articles, tokenizer, max_len=128, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.mlm_prob  = mlm_prob

        self.all_sentences = []
        for art in articles:
            sents = [s.strip() for s in art.split('.') if len(s.strip()) > 10]
            if len(sents) >= 2:
                self.all_sentences.append(sents)

        self.pairs = []
        for sents in self.all_sentences:
            for i in range(len(sents) - 1):
                sent_a = sents[i]
                if random.random() < 0.5:
                    sent_b = sents[i + 1]
                    nsp_lbl = 0
                else:
                    other_doc = random.choice(self.all_sentences)
                    sent_b = random.choice(other_doc)
                    nsp_lbl = 1

                self.pairs.append((sent_a, sent_b, nsp_lbl))

    def _mask_tokens(self, input_ids):
        labels = [-100] * len(input_ids)
        mask_tok = self.tokenizer.mask_token_id
        vocab_size = self.tokenizer.vocab_size
        special_ids = set(self.tokenizer.all_special_ids)

        for i, tok in enumerate(input_ids):
            if tok in special_ids:
                continue
            if random.random() < self.mlm_prob:
                labels[i] = tok
                r = random.random()
                if r < 0.80:
                    input_ids[i] = mask_tok
                elif r < 0.90:
                    input_ids[i] = random.randint(0, vocab_size - 1)

        return input_ids, labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sent_a, sent_b, nsp_label = self.pairs[idx]

        enc = self.tokenizer(
            sent_a, sent_b,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )

        input_ids, mlm_labels = self._mask_tokens(enc["input_ids"])

        return {
            "input_ids": torch.tensor(input_ids),
            "token_type_ids": torch.tensor(enc["token_type_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "mlm_labels": torch.tensor(mlm_labels),
            "nsp_label": torch.tensor(nsp_label),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train(num_epochs=3, batch_size=16, lr=1e-4, max_len=128):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    os.makedirs("artifacts/bert_base", exist_ok=True)

    # DATA
    raw = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    dataset = BERTPretrainDataset(raw["article"], tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # MODEL
    cfg = BERT_config(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        num_layers=12,
        n_heads=12,
        max_position_embeddings=max_len,
        type_vocab_size=2,
    )

    model = BERTModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ── metrics storage ──
    epoch_losses = []
    mlm_losses = []
    nsp_losses = []
    mlm_accuracies = []
    nsp_accuracies = []

    # ── TRAIN LOOP ──
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_mlm_loss = 0
        total_nsp_loss = 0

        mlm_correct = mlm_total = 0
        nsp_correct = nsp_total = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):

            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            nsp_label = batch["nsp_label"].to(device)

            loss, mlm_logits, nsp_logits, _, _ = model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
                nsp_label=nsp_label,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # MLM loss
            mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = mlm_loss_fn(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1)
            )
            total_mlm_loss += mlm_loss.item()

            # NSP loss
            nsp_loss_fn = nn.CrossEntropyLoss()
            nsp_loss = nsp_loss_fn(nsp_logits, nsp_label)
            total_nsp_loss += nsp_loss.item()

            # ── ACCURACY ──
            with torch.no_grad():
                # MLM accuracy
                preds = mlm_logits.argmax(dim=-1)
                mask = mlm_labels != -100
                mlm_correct += (preds[mask] == mlm_labels[mask]).sum().item()
                mlm_total += mask.sum().item()

                # NSP accuracy
                nsp_preds = nsp_logits.argmax(dim=-1)
                nsp_correct += (nsp_preds == nsp_label).sum().item()
                nsp_total += nsp_label.size(0)

        # ── epoch metrics ──
        epoch_losses.append(total_loss / len(dataloader))
        mlm_losses.append(total_mlm_loss / len(dataloader))
        nsp_losses.append(total_nsp_loss / len(dataloader))
        ckpt_path = f"artifacts/bert_base/bert_epoch_{epoch+1}.pt"

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.__dict__,
        }, ckpt_path)

        print(f"✅ Checkpoint saved → {ckpt_path}")
        mlm_accuracies.append(mlm_correct / mlm_total)
        nsp_accuracies.append(nsp_correct / nsp_total)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {epoch_losses[-1]:.4f}")
        print(f"MLM Acc: {mlm_accuracies[-1]:.4f}")
        print(f"NSP Acc: {nsp_accuracies[-1]:.4f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SAVE METRICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    torch.save({
        "loss": epoch_losses,
        "mlm_loss": mlm_losses,
        "nsp_loss": nsp_losses,
        "mlm_acc": mlm_accuracies,
        "nsp_acc": nsp_accuracies
    }, "artifacts/bert_base/metrics.pt")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PLOTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # LOSS PLOT
    plt.figure()
    plt.plot(epoch_losses, label="Total Loss")
    plt.plot(mlm_losses, label="MLM Loss")
    plt.plot(nsp_losses, label="NSP Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("artifacts/bert_base/loss.png")

    # ACCURACY PLOT
    plt.figure()
    plt.plot(mlm_accuracies, label="MLM Accuracy")
    plt.plot(nsp_accuracies, label="NSP Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("artifacts/bert_base/accuracy.png")

    print("\n✅ Charts saved in artifacts/bert_base/")


if __name__ == "__main__":
    train(num_epochs=1) 