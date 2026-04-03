import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast
from config_bert import *
from BERT import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== 1. DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== 2. LOAD DATASET =====
dataset = load_dataset("imdb")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    enc = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc["token_type_ids"],
        "labels": example["label"]
    }

dataset = dataset.map(tokenize_fn, batched=True)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
)

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=16)

# ===== 3. LOAD YOUR BERT =====
ckpt = torch.load("bert_pretrain_epoch3.pt", map_location=device)

cfg = BERT_config()
cfg.__dict__.update(ckpt["config"])

bert = BERTModel(cfg)
bert.load_state_dict(ckpt["model_state"])

# ===== 4. CLASSIFIER =====
class BERTForClassification(nn.Module):
    def __init__(self, bert, num_classes):
        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(cfg.d_model, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, _, _, _, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask
        )
        return self.fc(pooled_output)

model = BERTForClassification(bert, num_classes=2).to(device)

# ===== 5. TRAIN SETUP =====
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 5

# ===== 6. TRACKING =====
train_losses = []
train_accuracies = []
def train():
# ===== 7. TRAIN LOOP =====
    for epoch in range(EPOCHS):
        model.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        total_loss = 0
        correct = 0
        total = 0

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, token_type_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total

            loop.set_postfix(loss=loss.item(), acc=acc)

        avg_loss = total_loss / len(train_loader)
        epoch_acc = correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(epoch_acc)

        print(f"\nEpoch {epoch+1} Done | Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.4f}")

 # ===== 8. SAVE MODEL =====
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__
    }, "artifacts/bert_sentiment.pt")

    print("Model saved in artifacts/")

    # ===== 9. PLOT GRAPHS =====
    # ===== SAVE LOSS GRAPH =====
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("artifacts/loss.png")
    plt.close()

    # ===== SAVE ACCURACY GRAPH =====
    plt.figure()
    plt.plot(train_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.savefig("artifacts/accuracy.png")
    plt.close()

    print("Graphs saved in artifacts/ ✅")

if __name__=="__main__":
    train()