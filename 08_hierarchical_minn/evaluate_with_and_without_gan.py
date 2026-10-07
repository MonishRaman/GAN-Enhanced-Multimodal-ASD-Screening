"""
evaluate_with_and_without_gan.py
Comprehensive Benchmarking of:
1. Logistic Regression
2. Random Forest
3. SVM-RBF
4. XGBoost
5. Proposed Hierarchical Attention MINN

Evaluated under two conditions:
- WITHOUT GAN (Original imbalanced genetic training data)
- WITH GAN (GAN-augmented balanced genetic training data)

Evaluated on the exact same held-out unseen test cohort.
Generates metrics CSV, comparison plots, and roc curves.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from models.hierarchical_minn import HierarchicalAttentionMINN

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_DIR = Path("d:/GAN-Enhanced-Multimodal-ASD-Screening")
MINN_DIR = PROJECT_DIR / "08_hierarchical_minn"
DATA_DIR = MINN_DIR / "data"
FIGURE_DIR = MINN_DIR / "figures"
RESULT_DIR = MINN_DIR / "results"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

class MultimodalDataset(Dataset):
    def __init__(self, g, b, e, y):
        self.g = torch.tensor(g, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.e = torch.tensor(e, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.g[idx], self.b[idx], self.e[idx], self.y[idx]

def extract_features(df, target_cols=["Target", "Class/ASD"]):
    label_col = next((c for c in target_cols if c in df.columns), None)
    feat_cols = [c for c in df.columns if c not in [label_col, "Sample_Index", "GSM_ID", "Is_Synthetic"]]
    X = df[feat_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)
    return X, y

def generate_triplets(G_data, yG, B_data, yB, E_data, yE, n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    G_list, B_list, E_list, y_list = [], [], [], []
    for cls in [0, 1]:
        g_idx = np.where(yG == cls)[0]
        b_idx = np.where(yB == cls)[0]
        e_idx = np.where(yE == cls)[0]
        g_s = rng.choice(g_idx, size=n_per_class, replace=True)
        b_s = rng.choice(b_idx, size=n_per_class, replace=True)
        e_s = rng.choice(e_idx, size=n_per_class, replace=True)
        G_list.append(G_data[g_s])
        B_list.append(B_data[b_s])
        E_list.append(E_data[e_s])
        y_list.append(np.full(n_per_class, cls, dtype=np.float32))

    G_all = np.concatenate(G_list, axis=0)
    B_all = np.concatenate(B_list, axis=0)
    E_all = np.concatenate(E_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    perm = rng.permutation(len(y_all))
    return G_all[perm], B_all[perm], E_all[perm], y_all[perm]

def prepare_multimodal_data(use_gan=True, n_train=5000, n_val=1500, n_test=2500):
    g_train_df = pd.read_csv(DATA_DIR / "genetic_32d_train_gan_balanced.csv")
    if not use_gan:
        # Filter out synthetic samples to get original imbalanced real training data (69 ASD : 23 Control)
        g_train_df = g_train_df[g_train_df["Is_Synthetic"] == 0].reset_index(drop=True)

    g_test_df = pd.read_csv(DATA_DIR / "genetic_32d_test_unseen.csv")
    b_all_df = pd.read_csv(DATA_DIR / "behavioral_32d_all.csv")
    e_all_df = pd.read_csv(DATA_DIR / "environmental_32d_all.csv")

    G_train_raw, yG_train_raw = extract_features(g_train_df)
    G_test, yG_test = extract_features(g_test_df)

    g_tr_idx, g_va_idx = train_test_split(
        np.arange(len(G_train_raw)), test_size=0.15, random_state=SEED, stratify=yG_train_raw
    )
    G_train, yG_train = G_train_raw[g_tr_idx], yG_train_raw[g_tr_idx]
    G_val, yG_val = G_train_raw[g_va_idx], yG_train_raw[g_va_idx]

    B_all, yB_all = extract_features(b_all_df)
    B_train, B_temp, yB_train, yB_temp = train_test_split(B_all, yB_all, test_size=0.30, random_state=SEED, stratify=yB_all)
    B_val, B_test, yB_val, yB_test = train_test_split(B_temp, yB_temp, test_size=0.50, random_state=SEED, stratify=yB_temp)

    E_all, yE_all = extract_features(e_all_df)
    E_train, E_temp, yE_train, yE_temp = train_test_split(E_all, yE_all, test_size=0.30, random_state=SEED, stratify=yE_all)
    E_val, E_test, yE_val, yE_test = train_test_split(E_temp, yE_temp, test_size=0.50, random_state=SEED, stratify=yE_temp)

    G_tr, B_tr, E_tr, y_tr = generate_triplets(G_train, yG_train, B_train, yB_train, E_train, yE_train, n_train, seed=42)
    G_va, B_va, E_va, y_va = generate_triplets(G_val, yG_val, B_val, yB_val, E_val, yE_val, n_val, seed=123)
    G_te, B_te, E_te, y_te = generate_triplets(G_test, yG_test, B_test, yB_test, E_test, yE_test, n_test, seed=2026)

    train_ds = MultimodalDataset(G_tr, B_tr, E_tr, y_tr)
    val_ds = MultimodalDataset(G_va, B_va, E_va, y_va)
    test_ds = MultimodalDataset(G_te, B_te, E_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    return (G_tr, B_tr, E_tr, y_tr), (G_va, B_va, E_va, y_va), (G_te, B_te, E_te, y_te), train_loader, val_loader, test_loader

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_preds = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_preds)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_preds)),
        "Precision": float(precision_score(y_true, y_preds, zero_division=0)),
        "Recall": float(recall_score(y_true, y_preds, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "F1_Score": float(f1_score(y_true, y_preds, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_probs)),
        "Brier_Score": float(brier_score_loss(y_true, y_probs)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def train_eval_minn(train_loader, val_loader, test_loader, epochs=25):
    model = HierarchicalAttentionMINN(in_features=32, branch_hidden=64, branch_out=32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for g, b, e, y in train_loader:
            g, b, e, y = g.to(DEVICE), b.to(DEVICE), e.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(g, b, e)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for g, b, e, y in val_loader:
                g, b, e, y = g.to(DEVICE), b.to(DEVICE), e.to(DEVICE), y.to(DEVICE)
                logits, _ = model(g, b, e)
                val_loss += criterion(logits, y).item() * len(y)
                val_samples += len(y)
        avg_val_loss = val_loss / val_samples
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    test_probs = []
    test_targets = []
    with torch.no_grad():
        for g, b, e, y in test_loader:
            g, b, e = g.to(DEVICE), b.to(DEVICE), e.to(DEVICE)
            _, probs = model(g, b, e)
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(y.numpy())

    return np.array(test_probs), np.array(test_targets), model

def run_benchmarks():
    print("=" * 70)
    print("RUNNING MULTIMODAL BENCHMARKS: WITH GAN VS WITHOUT GAN")
    print("Models: Logistic Regression, Random Forest, SVM-RBF, XGBoost, Proposed MINN")
    print("=" * 70)

    results = []
    roc_curves_data = {}

    for condition in ["Without_GAN", "With_GAN"]:
        use_gan = (condition == "With_GAN")
        print(f"\n>>> Running for Condition: {condition.upper()} (use_gan={use_gan}) <<<")

        (G_tr, B_tr, E_tr, y_tr), _, (G_te, B_te, E_te, y_te), tr_loader, va_loader, te_loader = prepare_multimodal_data(use_gan=use_gan)
        X_tr = np.concatenate([G_tr, B_tr, E_tr], axis=1) # 96-D multimodal representation
        X_te = np.concatenate([G_te, B_te, E_te], axis=1)

        # 1. Logistic Regression
        print("Training 1. Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")
        lr.fit(X_tr, y_tr)
        lr_probs = lr.predict_proba(X_te)[:, 1]
        m_lr = compute_metrics(y_te, lr_probs)
        results.append({"Model": "Logistic Regression", "Condition": condition, **m_lr})
        roc_curves_data[f"Logistic Regression ({condition})"] = (y_te, lr_probs)

        # 2. Random Forest
        print("Training 2. Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_probs = rf.predict_proba(X_te)[:, 1]
        m_rf = compute_metrics(y_te, rf_probs)
        results.append({"Model": "Random Forest", "Condition": condition, **m_rf})
        roc_curves_data[f"Random Forest ({condition})"] = (y_te, rf_probs)

        # 3. SVM-RBF
        print("Training 3. SVM-RBF...")
        # Subsample for faster SVM training if needed (SVM O(N^2))
        sub_idx = np.random.choice(len(X_tr), size=min(4000, len(X_tr)), replace=False)
        svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)
        svm.fit(X_tr[sub_idx], y_tr[sub_idx])
        svm_probs = svm.predict_proba(X_te)[:, 1]
        m_svm = compute_metrics(y_te, svm_probs)
        results.append({"Model": "SVM-RBF", "Condition": condition, **m_svm})
        roc_curves_data[f"SVM-RBF ({condition})"] = (y_te, svm_probs)

        # 4. XGBoost
        print("Training 4. XGBoost...")
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=SEED, eval_metric="logloss")
        xgb.fit(X_tr, y_tr)
        xgb_probs = xgb.predict_proba(X_te)[:, 1]
        m_xgb = compute_metrics(y_te, xgb_probs)
        results.append({"Model": "XGBoost", "Condition": condition, **m_xgb})
        roc_curves_data[f"XGBoost ({condition})"] = (y_te, xgb_probs)

        # 5. Proposed Hierarchical Attention MINN
        print(f"Training 5. Proposed Hierarchical Attention MINN ({condition})...")
        if use_gan:
            # Check if saved minn_best.pt exists
            minn_path = MINN_DIR / "models" / "minn_best.pt"
            if minn_path.exists():
                minn_model = HierarchicalAttentionMINN(in_features=32, branch_hidden=64, branch_out=32).to(DEVICE)
                minn_model.load_state_dict(torch.load(minn_path, map_location=DEVICE))
                minn_model.eval()
                test_probs = []
                with torch.no_grad():
                    for g, b, e, y in te_loader:
                        g, b, e = g.to(DEVICE), b.to(DEVICE), e.to(DEVICE)
                        _, probs = minn_model(g, b, e)
                        test_probs.extend(probs.cpu().numpy())
                minn_probs = np.array(test_probs)
            else:
                minn_probs, _, _ = train_eval_minn(tr_loader, va_loader, te_loader, epochs=25)
        else:
            minn_probs, _, _ = train_eval_minn(tr_loader, va_loader, te_loader, epochs=25)

        m_minn = compute_metrics(y_te, minn_probs)
        results.append({"Model": "Proposed Hierarchical Attention MINN", "Condition": condition, **m_minn})
        roc_curves_data[f"Proposed MINN ({condition})"] = (y_te, minn_probs)

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULT_DIR / "minn_vs_other_models_detailed.csv", index=False)
    print("\nBENCHMARK RESULTS TABLE:")
    print(res_df[["Model", "Condition", "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]].to_string(index=False))

    # Generate Figures
    generate_comparison_figures(res_df, roc_curves_data)

    return res_df

def generate_comparison_figures(res_df, roc_curves_data):
    # Figure 1: Comprehensive ROC Curves (With & Without GAN)
    plt.figure(figsize=(10, 8))
    palette = {
        "Proposed MINN (With_GAN)": ("#DC2626", 2.8, "-"),
        "Proposed MINN (Without_GAN)": ("#EF4444", 2.0, "--"),
        "XGBoost (With_GAN)": ("#059669", 2.0, "-"),
        "XGBoost (Without_GAN)": ("#34D399", 1.5, "--"),
        "Random Forest (With_GAN)": ("#2563EB", 2.0, "-"),
        "Random Forest (Without_GAN)": ("#60A5FA", 1.5, "--"),
        "SVM-RBF (With_GAN)": ("#7C3AED", 2.0, "-"),
        "SVM-RBF (Without_GAN)": ("#A78BFA", 1.5, "--"),
        "Logistic Regression (With_GAN)": ("#D97706", 1.8, "-"),
        "Logistic Regression (Without_GAN)": ("#FBBF24", 1.5, "--"),
    }

    for label, (y_true, probs) in roc_curves_data.items():
        if label in palette:
            color, lw, ls = palette[label]
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            plt.plot(fpr, tpr, color=color, lw=lw, linestyle=ls, label=f"{label} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("ROC Curve Benchmarking: Proposed MINN vs. Baseline Models\n(Impact of GAN Augmentation)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1_path = FIGURE_DIR / "minn_comparison_roc_curves.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig1_path}")

    # Figure 2: Grouped Bar Chart of Accuracy, F1-Score & ROC-AUC
    models = ["Logistic Regression", "Random Forest", "SVM-RBF", "XGBoost", "Proposed MINN"]
    
    # Shorten MINN label in df for plotting
    plot_df = res_df.copy()
    plot_df["Model_Short"] = plot_df["Model"].replace({"Proposed Hierarchical Attention MINN": "Proposed MINN"})

    plt.figure(figsize=(12, 6))
    metrics_to_plot = ["Accuracy", "F1_Score", "ROC_AUC"]
    x = np.arange(len(models))
    width = 0.13

    colors = {
        ("Without_GAN", "Accuracy"): "#93C5FD",
        ("With_GAN", "Accuracy"): "#1D4ED8",
        ("Without_GAN", "F1_Score"): "#A7F3D0",
        ("With_GAN", "F1_Score"): "#047857",
        ("Without_GAN", "ROC_AUC"): "#FDE68A",
        ("With_GAN", "ROC_AUC"): "#D97706",
    }

    fig, ax = plt.subplots(figsize=(13, 6.5))

    offsets = [-0.3, -0.18, -0.06, 0.06, 0.18, 0.3]
    bars_labels = [
        ("Without_GAN", "Accuracy", "Acc (No GAN)"),
        ("With_GAN", "Accuracy", "Acc (With GAN)"),
        ("Without_GAN", "F1_Score", "F1 (No GAN)"),
        ("With_GAN", "F1_Score", "F1 (With GAN)"),
        ("Without_GAN", "ROC_AUC", "AUC (No GAN)"),
        ("With_GAN", "ROC_AUC", "AUC (With GAN)"),
    ]

    for idx, (cond, met, lbl) in enumerate(bars_labels):
        vals = []
        for m in models:
            row = plot_df[(plot_df["Model_Short"] == m) & (plot_df["Condition"] == cond)]
            vals.append(row[met].values[0] if len(row) > 0 else 0)
        ax.bar(x + offsets[idx], vals, width=0.10, label=lbl, color=colors[(cond, met)], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    ax.set_title("Multimodal ASD Screening: Performance Comparison Across Models (With vs. Without GAN)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig2_path = FIGURE_DIR / "minn_comparison_metrics_barchart.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig2_path}")

    # Figure 3: Delta / Gain from GAN
    piv = plot_df.pivot(index="Model_Short", columns="Condition", values=["Accuracy", "F1_Score", "ROC_AUC"])
    delta_df = pd.DataFrame({
        "Accuracy Gain (%)": (piv[("Accuracy", "With_GAN")] - piv[("Accuracy", "Without_GAN")]) * 100,
        "F1-Score Gain (%)": (piv[("F1_Score", "With_GAN")] - piv[("F1_Score", "Without_GAN")]) * 100,
        "ROC-AUC Gain (%)": (piv[("ROC_AUC", "With_GAN")] - piv[("ROC_AUC", "Without_GAN")]) * 100,
    }).reindex(models)

    plt.figure(figsize=(10, 5.5))
    ax = delta_df.plot(kind="bar", figsize=(10, 5.5), colormap="viridis", edgecolor="black", linewidth=0.6)
    plt.axhline(0, color="black", linewidth=1.2, linestyle="--")
    plt.title("Net Impact of Objective 2 GAN Augmentation Across Models (Percentage Gain)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Percentage Point Change (%)", fontsize=11, fontweight="bold")
    plt.xticks(ticks=range(len(models)), labels=models, rotation=0, fontsize=10, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig3_path = FIGURE_DIR / "minn_gan_gain_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig3_path}")

if __name__ == "__main__":
    run_benchmarks()
