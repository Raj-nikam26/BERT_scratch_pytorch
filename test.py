import torch
import torch.nn as nn
from transformers import BertTokenizer
from config_bert import *
from finetune_for_classification import BERTForClassification
from BERT import BERTModel
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. LOAD CHECKPOINT =====
ckpt = torch.load("artifacts/bert_sentiment.pt", map_location=device)

# ===== 3. LOAD CONFIG =====
cfg = BERT_config()
cfg.__dict__.update(ckpt["config"])
bert=BERTModel(cfg)

model=BERTForClassification(bert,2).to(device)
model.load_state_dict(ckpt["model_state"], strict=False)

model.eval()

# ===== 6. PREDICT FUNCTION =====
def predict(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc["token_type_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs.cpu().numpy()

# ===== 7. TEST =====
text = "This movie was not bad"
pred, probs = predict(text)

print("Text:", text)
print("Prediction:", "Positive" if pred == 1 else "Negative")
print("Probabilities:", probs)