import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import random
from model_parts import EncoderStack, LayerNorm
from config_bert import BERT_config
from BERT import BERTModel
from torch.utils.data import Dataset, DataLoader  
os.makedirs("artifacts/bert_distilled", exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DISTILLATION LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    - Distillation loss (KL divergence between teacher and student softmax outputs)
    - MLM loss (student's own MLM predictions)
    - NSP loss (student's own NSP predictions)
    """
    def __init__(self, temperature=3.0, alpha=0.7):
        """
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss (1-alpha for student loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, student_mlm_logits, student_nsp_logits, 
                teacher_mlm_logits, teacher_nsp_logits,
                mlm_labels, nsp_labels):
        """
        Compute total distillation loss
        
        Args:
            student_mlm_logits: Student's MLM logits [batch_size, seq_len, vocab_size]
            student_nsp_logits: Student's NSP logits [batch_size, 2]
            teacher_mlm_logits: Teacher's MLM logits [batch_size, seq_len, vocab_size]
            teacher_nsp_logits: Teacher's NSP logits [batch_size, 2]
            mlm_labels: Ground truth MLM labels
            nsp_labels: Ground truth NSP labels
        """
        # 1. Distillation Loss (soft targets)
        # MLM distillation
        student_mlm_soft = F.log_softmax(student_mlm_logits / self.temperature, dim=-1)
        teacher_mlm_soft = F.softmax(teacher_mlm_logits / self.temperature, dim=-1)
        
        # Only compute distillation loss on masked positions
        mask = (mlm_labels != -100).unsqueeze(-1).float()
        masked_student_mlm_soft = student_mlm_soft * mask
        masked_teacher_mlm_soft = teacher_mlm_soft * mask
        
        mlm_distill_loss = self.kl_div(
            masked_student_mlm_soft.view(-1, student_mlm_soft.size(-1)),
            masked_teacher_mlm_soft.view(-1, teacher_mlm_soft.size(-1))
        )
        
        # NSP distillation
        student_nsp_soft = F.log_softmax(student_nsp_logits / self.temperature, dim=-1)
        teacher_nsp_soft = F.softmax(teacher_nsp_logits / self.temperature, dim=-1)
        nsp_distill_loss = self.kl_div(student_nsp_soft, teacher_nsp_soft)
        
        distill_loss = mlm_distill_loss + nsp_distill_loss
        
        # 2. Student's own supervised loss (hard targets)
        student_mlm_loss = self.mlm_loss_fn(
            student_mlm_logits.view(-1, student_mlm_logits.size(-1)),
            mlm_labels.view(-1)
        )
        student_nsp_loss = self.nsp_loss_fn(student_nsp_logits, nsp_labels)
        student_loss = student_mlm_loss + student_nsp_loss
        
        # Combine losses
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return total_loss, student_mlm_loss, student_nsp_loss, distill_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET (unchanged from your original)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BERTPretrainDataset(Dataset):
    def __init__(self, articles, tokenizer, max_len=128, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob

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


# DISTILLATION TRAINER

def train_distillation(teacher_checkpoint_path, num_epochs=3, batch_size=16, 
                       lr=1e-4, max_len=128, temperature=3.0, alpha=0.7):
    """
    Train student model using knowledge distillation from teacher model
    
    Args:
        teacher_checkpoint_path: Path to pre-trained teacher model checkpoint
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_len: Maximum sequence length
        temperature: Temperature for softening probabilities
        alpha: Weight for distillation loss
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Load dataset
    print("Loading dataset...")
    raw = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]")
    dataset = BERTPretrainDataset(raw["article"], tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}")


    # TEACHER MODEL (Large, pre-trained)

    print("Loading teacher model...")
    teacher_cfg = BERT_config(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        num_layers=12,
        n_heads=12,
        max_position_embeddings=max_len,
        type_vocab_size=2,
    )
    
    teacher_model = BERTModel(teacher_cfg).to(device)
    
    # Load pre-trained teacher checkpoint
    if os.path.exists(teacher_checkpoint_path):
        checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
        teacher_model.load_state_dict(checkpoint["model_state"])
        print(f"Teacher model loaded from {teacher_checkpoint_path}")
    else:
        print(f"Warning: Teacher checkpoint not found at {teacher_checkpoint_path}")
        print("Using randomly initialized teacher model")
    
    teacher_model.eval()  # Set teacher to evaluation mode
    for param in teacher_model.parameters():
        param.requires_grad = False  # Freeze teacher parameters
    

    # STUDENT MODEL
 
    print("Creating student model (smaller architecture)...")
    student_cfg = BERT_config(
        vocab_size=tokenizer.vocab_size,
        d_model=384,           # Half of teacher's hidden size
        num_layers=4,          # Fewer layers
        n_heads=6,             # Fewer attention heads
        max_position_embeddings=max_len,
        type_vocab_size=2,
    )
    
    student_model = BERTModel(student_cfg).to(device)
    
    # Print model size comparison
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"\nModel Size Comparison:")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.1f}x")
    
    # Optimizer for student
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)
    
    # Distillation loss
    distill_loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)
    
    # Metrics storage
    epoch_losses = []
    student_mlm_losses = []
    student_nsp_losses = []
    distill_losses = []
    mlm_accuracies = []
    nsp_accuracies = []

    # TRAINING LOOP

    for epoch in range(num_epochs):
        student_model.train()
        
        total_loss = 0
        total_student_mlm_loss = 0
        total_student_nsp_loss = 0
        total_distill_loss = 0
        
        mlm_correct = mlm_total = 0
        nsp_correct = nsp_total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            nsp_label = batch["nsp_label"].to(device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                _, teacher_mlm_logits, teacher_nsp_logits, _, _ = teacher_model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    mlm_labels=mlm_labels,
                    nsp_label=nsp_label,
                )
            
            # Get student predictions
            _, student_mlm_logits, student_nsp_logits, _, _ = student_model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
                nsp_label=nsp_label,
            )
            
            # Compute distillation loss
            loss, student_mlm_loss, student_nsp_loss, distill_loss = distill_loss_fn(
                student_mlm_logits, student_nsp_logits,
                teacher_mlm_logits, teacher_nsp_logits,
                mlm_labels, nsp_label
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_student_mlm_loss += student_mlm_loss.item()
            total_student_nsp_loss += student_nsp_loss.item()
            total_distill_loss += distill_loss.item()
            
            # Accuracy metrics
            with torch.no_grad():
                # MLM accuracy
                preds = student_mlm_logits.argmax(dim=-1)
                mask = mlm_labels != -100
                mlm_correct += (preds[mask] == mlm_labels[mask]).sum().item()
                mlm_total += mask.sum().item()
                
                # NSP accuracy
                nsp_preds = student_nsp_logits.argmax(dim=-1)
                nsp_correct += (nsp_preds == nsp_label).sum().item()
                nsp_total += nsp_label.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mlm_acc': f'{mlm_correct/max(mlm_total, 1):.3f}',
                'nsp_acc': f'{nsp_correct/max(nsp_total, 1):.3f}'
            })
        
        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        avg_student_mlm_loss = total_student_mlm_loss / len(dataloader)
        avg_student_nsp_loss = total_student_nsp_loss / len(dataloader)
        avg_distill_loss = total_distill_loss / len(dataloader)
        
        mlm_acc = mlm_correct / mlm_total
        nsp_acc = nsp_correct / nsp_total
        
        epoch_losses.append(avg_loss)
        student_mlm_losses.append(avg_student_mlm_loss)
        student_nsp_losses.append(avg_student_nsp_loss)
        distill_losses.append(avg_distill_loss)
        mlm_accuracies.append(mlm_acc)
        nsp_accuracies.append(nsp_acc)
        
        # Save checkpoint
        checkpoint_path = f"artifacts/bert_distilled/student_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state": student_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": student_cfg.__dict__,
            "teacher_config": teacher_cfg.__dict__,
            "temperature": temperature,
            "alpha": alpha,
            "metrics": {
                "loss": avg_loss,
                "mlm_acc": mlm_acc,
                "nsp_acc": nsp_acc,
                "student_mlm_loss": avg_student_mlm_loss,
                "student_nsp_loss": avg_student_nsp_loss,
                "distill_loss": avg_distill_loss
            }
        }, checkpoint_path)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"Total Loss: {avg_loss:.4f}")
        print(f"  - Student MLM Loss: {avg_student_mlm_loss:.4f}")
        print(f"  - Student NSP Loss: {avg_student_nsp_loss:.4f}")
        print(f"  - Distillation Loss: {avg_distill_loss:.4f}")
        print(f"MLM Accuracy: {mlm_acc:.4f}")
        print(f"NSP Accuracy: {nsp_acc:.4f}")
        print(f"✅ Checkpoint saved → {checkpoint_path}")
        print(f"{'='*60}\n")

    # SAVE METRICS AND PLOTS

    
    # Save metrics
    metrics = {
        "loss": epoch_losses,
        "student_mlm_loss": student_mlm_losses,
        "student_nsp_loss": student_nsp_losses,
        "distill_loss": distill_losses,
        "mlm_acc": mlm_accuracies,
        "nsp_acc": nsp_accuracies,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / student_params,
        "temperature": temperature,
        "alpha": alpha
    }
    torch.save(metrics, "artifacts/bert_distilled/metrics.pt")
    
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label="Total Loss", linewidth=2)
    plt.plot(student_mlm_losses, label="Student MLM Loss", linestyle='--')
    plt.plot(student_nsp_losses, label="Student NSP Loss", linestyle='--')
    plt.plot(distill_losses, label="Distillation Loss", linestyle=':')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(mlm_accuracies, label="MLM Accuracy", linewidth=2, marker='o')
    plt.plot(nsp_accuracies, label="NSP Accuracy", linewidth=2, marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Student Model Accuracy")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("artifacts/bert_distilled/training_metrics.png", dpi=150)
    plt.show()
    
    # Save model configuration
    import json
    config_info = {
        "student_config": student_cfg.__dict__,
        "teacher_config": teacher_cfg.__dict__,
        "training_config": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_len": max_len,
            "temperature": temperature,
            "alpha": alpha
        }
    }
    with open("artifacts/bert_distilled/config.json", "w") as f:
        json.dump(config_info, f, indent=2)
    
    print("\n✅ Training completed!")
    print(f"📊 Metrics saved in artifacts/bert_distilled/")
    print(f"🎯 Final Student MLM Accuracy: {mlm_accuracies[-1]:.4f}")
    print(f"🎯 Final Student NSP Accuracy: {nsp_accuracies[-1]:.4f}")



# EVALUATION FUNCTION

def evaluate_distilled_model(student_checkpoint_path, teacher_checkpoint_path, 
                             num_samples=1000):
    """
    Compare student and teacher model performance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Load teacher
    print("Loading teacher model...")
    teacher_cfg = BERT_config(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        num_layers=12,
        n_heads=12,
        max_position_embeddings=128,
        type_vocab_size=2,
    )
    teacher_model = BERTModel(teacher_cfg).to(device)
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    teacher_model.load_state_dict(teacher_checkpoint["model_state"])
    teacher_model.eval()
    
    # Load student
    print("Loading student model...")
    student_checkpoint = torch.load(student_checkpoint_path, map_location=device)
    student_cfg = BERT_config(**student_checkpoint["config"])
    student_model = BERTModel(student_cfg).to(device)
    student_model.load_state_dict(student_checkpoint["model_state"])
    student_model.eval()
    
    # Load evaluation data
    raw = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{num_samples}]")
    dataset = BERTPretrainDataset(raw["article"], tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    teacher_mlm_correct = teacher_mlm_total = 0
    teacher_nsp_correct = teacher_nsp_total = 0
    student_mlm_correct = student_mlm_total = 0
    student_nsp_correct = student_nsp_total = 0
    
    print("\nEvaluating models...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            nsp_label = batch["nsp_label"].to(device)
            
            # Teacher evaluation
            _, teacher_mlm_logits, teacher_nsp_logits, _, _ = teacher_model(
                input_ids, token_type_ids, attention_mask, mlm_labels, nsp_label
            )
            
            teacher_mlm_preds = teacher_mlm_logits.argmax(dim=-1)
            teacher_mask = mlm_labels != -100
            teacher_mlm_correct += (teacher_mlm_preds[teacher_mask] == mlm_labels[teacher_mask]).sum().item()
            teacher_mlm_total += teacher_mask.sum().item()
            
            teacher_nsp_preds = teacher_nsp_logits.argmax(dim=-1)
            teacher_nsp_correct += (teacher_nsp_preds == nsp_label).sum().item()
            teacher_nsp_total += nsp_label.size(0)
            
            # Student evaluation
            _, student_mlm_logits, student_nsp_logits, _, _ = student_model(
                input_ids, token_type_ids, attention_mask, mlm_labels, nsp_label
            )
            
            student_mlm_preds = student_mlm_logits.argmax(dim=-1)
            student_mlm_correct += (student_mlm_preds[teacher_mask] == mlm_labels[teacher_mask]).sum().item()
            student_mlm_total += teacher_mask.sum().item()
            
            student_nsp_preds = student_nsp_logits.argmax(dim=-1)
            student_nsp_correct += (student_nsp_preds == nsp_label).sum().item()
            student_nsp_total += nsp_label.size(0)
    
    teacher_mlm_acc = teacher_mlm_correct / teacher_mlm_total
    teacher_nsp_acc = teacher_nsp_correct / teacher_nsp_total
    student_mlm_acc = student_mlm_correct / student_mlm_total
    student_nsp_acc = student_nsp_correct / student_nsp_total
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Teacher Model:")
    print(f"  - MLM Accuracy: {teacher_mlm_acc:.4f}")
    print(f"  - NSP Accuracy: {teacher_nsp_acc:.4f}")
    print(f"\nStudent Model (Distilled):")
    print(f"  - MLM Accuracy: {student_mlm_acc:.4f}")
    print(f"  - NSP Accuracy: {student_nsp_acc:.4f}")
    print(f"\nPerformance Retention:")
    print(f"  - MLM: {student_mlm_acc/teacher_mlm_acc*100:.2f}% of teacher")
    print(f"  - NSP: {student_nsp_acc/teacher_nsp_acc*100:.2f}% of teacher")
    print("="*60)
    
    return {
        "teacher_mlm_acc": teacher_mlm_acc,
        "teacher_nsp_acc": teacher_nsp_acc,
        "student_mlm_acc": student_mlm_acc,
        "student_nsp_acc": student_nsp_acc
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # Path to pre-trained teacher model 
    teacher_checkpoint = "artifacts/bert_base/bert_epoch_1.pt"  # Adjusted based on  training
    student_checkpoint = f"artifacts/bert_distilled/student_epoch_{3}.pt"
    # Train distilled student model
    train_distillation(
        teacher_checkpoint_path=teacher_checkpoint,
        num_epochs=2,           # fewer epochs for distillation
        batch_size=16,
        lr=2e-4,                # Slightly higher learning rate for student
        max_len=128,
        temperature=3.0,        # Higher temperature = softer probabilities
        alpha=0.7              # 70% weight on distillation loss, 30% on student actual loss
    )
    

    evaluate_distilled_model(
        student_checkpoint_path="artifacts/bert_distilled/student_epoch_2.pt",
        teacher_checkpoint_path=teacher_checkpoint,
        num_samples=1000
    )