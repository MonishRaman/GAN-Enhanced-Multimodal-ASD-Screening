"""
Training Script for Objective 3 Hierarchical Attention-Based MINN.

Trains the Multi-Input Neural Network on the GAN-augmented balanced dataset.
Records training and validation loss, accuracy, F1, and ROC-AUC per epoch.
Saves the best model checkpoint and training curves.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from models.hierarchical_minn import HierarchicalAttentionMINN
from data_loader import get_multimodal_dataloaders

# Configuration
SEED = 42
EPOCHS = 35
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"
FIGURE_DIR = PROJECT_DIR / "figures"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []

    for g, b, e, y in loader:
        g, b, e, y = g.to(DEVICE), b.to(DEVICE), e.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        logits, probs = model(g, b, e)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend((probs.detach().cpu().numpy() >= 0.5).astype(int))
        all_targets.extend(y.cpu().numpy().astype(int))

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    return avg_loss, acc, f1, auc

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for g, b, e, y in loader:
            g, b, e, y = g.to(DEVICE), b.to(DEVICE), e.to(DEVICE), y.to(DEVICE)
            logits, probs = model(g, b, e)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs.cpu().numpy() >= 0.5).astype(int))
            all_targets.extend(y.cpu().numpy().astype(int))

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    return avg_loss, acc, f1, auc

def main():
    print("=" * 70)
    print("TRAINING OBJECTIVE 3: HIERARCHICAL ATTENTION-BASED MINN")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load dataloaders
    train_loader, val_loader, test_loader, _ = get_multimodal_dataloaders(
        batch_size=BATCH_SIZE,
        n_train=5000,
        n_val=1500,
        n_test=2500
    )

    model = HierarchicalAttentionMINN(in_features=32, branch_hidden=64, branch_out=32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_auc = 0.0
    patience_counter = 0
    history = []

    print("\nStarting training loop...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, tr_auc = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_f1, va_auc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1, "train_auc": tr_auc,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1, "val_auc": va_auc,
            "lr": scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch:02d}/{EPOCHS} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f}, AUC: {tr_auc:.4f} | Val Loss: {va_loss:.4f}, AUC: {va_auc:.4f}, Acc: {va_acc:.4f}, F1: {va_f1:.4f}")

        # Checkpoint best model based on Val AUC
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "minn_best.pt")
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
                break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULT_DIR / "training_history.csv", index=False)
    print(f"\n[OK] Training history saved to {RESULT_DIR / 'training_history.csv'}")

    # Save configuration
    config = {
        "model_name": "HierarchicalAttentionMINN",
        "objective": "Objective 3: Hierarchical Attention Multimodal Feature Fusion",
        "inputs": {"genetic_dim": 32, "behavioral_dim": 32, "environmental_dim": 32},
        "branch_architecture": "32 -> 64 -> 32 (BatchNorm + GELU + Dropout)",
        "attention_levels": {
            "level_1_intra_modality": "Feature-level softmax attention per modality",
            "level_2_inter_modality": "Dynamic per-patient softmax attention over [Genetic, Behavioral, Environmental]"
        },
        "fusion_type": "Adaptive Attention-Weighted Concatenation (96-D)",
        "classifier": "96 -> 48 -> 16 -> 1 with Dropout & Sigmoid",
        "training_data": "GAN-Augmented Aligned 32-D Balanced Dataset",
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc)
    }
    with open(RESULT_DIR / "minn_config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    axes[0].set_title("MINN Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_auc"], label="Train ROC-AUC", color="royalblue", lw=2)
    axes[1].plot(history_df["epoch"], history_df["val_auc"], label="Val ROC-AUC", color="crimson", lw=2, linestyle="--")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Accuracy", color="seagreen", lw=1.5, linestyle=":")
    axes[1].set_title("MINN Validation Metrics", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig_path = FIGURE_DIR / "training_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OK] Training curves plot saved to {fig_path}")

if __name__ == "__main__":
    main()
