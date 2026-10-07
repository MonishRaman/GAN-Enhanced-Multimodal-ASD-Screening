"""
Objective 3 Data Loader and Balanced Multimodal Generator.

Integrates Objective 2 GAN-augmented balanced genetic data with
behavioral and environmental representations.
Ensures strict featurization ordering and zero data leakage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SEED = 42

class MultimodalASDDataset(Dataset):
    """PyTorch Dataset for Multimodal ASD Data (Genetic, Behavioral, Environmental)."""
    def __init__(self, g_tensor, b_tensor, e_tensor, y_tensor):
        self.g = g_tensor
        self.b = b_tensor
        self.e = e_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.g[idx], self.b[idx], self.e[idx], self.y[idx]

def extract_features(df, target_col_names=["Target", "Class/ASD"]):
    label_col = None
    for name in target_col_names:
        if name in df.columns:
            label_col = name
            break
    if label_col is None:
        raise ValueError(f"Target column not found among {target_col_names}")

    # Feature columns are the 32 numeric columns
    feat_cols = [c for c in df.columns if c not in [label_col, "Sample_Index", "GSM_ID", "Is_Synthetic"]]
    assert len(feat_cols) == 32, f"Expected 32 feature columns, found {len(feat_cols)}"
    
    X = df[feat_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)
    return X, y

def generate_class_conditional_triplets(G_data, yG, B_data, yB, E_data, yE, samples_per_class, seed=42):
    """
    Constructs class-conditional multimodal triplets from the modality distributions.
    For each diagnostic class, G, B, and E are sampled independently from that class.
    """
    rng = np.random.default_rng(seed)
    G_parts, B_parts, E_parts, y_parts = [], [], [], []

    for cls in [0, 1]:
        g_idx = np.where(yG == cls)[0]
        b_idx = np.where(yB == cls)[0]
        e_idx = np.where(yE == cls)[0]

        if len(g_idx) == 0 or len(b_idx) == 0 or len(e_idx) == 0:
            raise ValueError(f"Class {cls} missing in one of the modalities.")

        g_sel = rng.choice(g_idx, size=samples_per_class, replace=True)
        b_sel = rng.choice(b_idx, size=samples_per_class, replace=True)
        e_sel = rng.choice(e_idx, size=samples_per_class, replace=True)

        G_parts.append(G_data[g_sel])
        B_parts.append(B_data[b_sel])
        E_parts.append(E_data[e_sel])
        y_parts.append(np.full(samples_per_class, cls, dtype=np.float32))

    G_all = np.concatenate(G_parts, axis=0)
    B_all = np.concatenate(B_parts, axis=0)
    E_all = np.concatenate(E_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    # Permute together
    perm = rng.permutation(len(y_all))
    return G_all[perm], B_all[perm], E_all[perm], y_all[perm]

def get_multimodal_dataloaders(data_dir=None, batch_size=64, n_train=5000, n_val=1500, n_test=2500):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    else:
        data_dir = Path(data_dir)

    # 1. Load GAN-balanced genetic training data & unseen genetic test data
    g_train_df = pd.read_csv(data_dir / "genetic_32d_train_gan_balanced.csv")
    g_test_df = pd.read_csv(data_dir / "genetic_32d_test_unseen.csv")
    
    # 2. Load behavioral and environmental data
    b_all_df = pd.read_csv(data_dir / "behavioral_32d_all.csv")
    e_all_df = pd.read_csv(data_dir / "environmental_32d_all.csv")

    # Extract features & labels
    G_train_raw, yG_train_raw = extract_features(g_train_df)
    G_test, yG_test = extract_features(g_test_df)

    # Split G_train_raw into train and val splits (preserving balance)
    g_tr_idx, g_va_idx = train_test_split(
        np.arange(len(G_train_raw)),
        test_size=0.15,
        random_state=SEED,
        stratify=yG_train_raw
    )
    G_train, yG_train = G_train_raw[g_tr_idx], yG_train_raw[g_tr_idx]
    G_val, yG_val = G_train_raw[g_va_idx], yG_train_raw[g_va_idx]

    # Split Behavioral (70% train, 15% val, 15% test)
    B_all, yB_all = extract_features(b_all_df)
    B_train, B_temp, yB_train, yB_temp = train_test_split(B_all, yB_all, test_size=0.30, random_state=SEED, stratify=yB_all)
    B_val, B_test, yB_val, yB_test = train_test_split(B_temp, yB_temp, test_size=0.50, random_state=SEED, stratify=yB_temp)

    # Split Environmental (70% train, 15% val, 15% test)
    E_all, yE_all = extract_features(e_all_df)
    E_train, E_temp, yE_train, yE_temp = train_test_split(E_all, yE_all, test_size=0.30, random_state=SEED, stratify=yE_all)
    E_val, E_test, yE_val, yE_test = train_test_split(E_temp, yE_temp, test_size=0.50, random_state=SEED, stratify=yE_temp)

    print(f"Modality Split Summary:")
    print(f"  Genetic (GAN-Balanced) Train: {len(G_train)}, Val: {len(G_val)}, Unseen Test: {len(G_test)}")
    print(f"  Behavioral             Train: {len(B_train)}, Val: {len(B_val)}, Test: {len(B_test)}")
    print(f"  Environmental          Train: {len(E_train)}, Val: {len(E_val)}, Test: {len(E_test)}")

    # Generate balanced triplets
    G_tr, B_tr, E_tr, y_tr = generate_class_conditional_triplets(G_train, yG_train, B_train, yB_train, E_train, yE_train, n_train, seed=42)
    G_va, B_va, E_va, y_va = generate_class_conditional_triplets(G_val, yG_val, B_val, yB_val, E_val, yE_val, n_val, seed=123)
    G_te, B_te, E_te, y_te = generate_class_conditional_triplets(G_test, yG_test, B_test, yB_test, E_test, yE_test, n_test, seed=2026)

    train_ds = MultimodalASDDataset(torch.tensor(G_tr), torch.tensor(B_tr), torch.tensor(E_tr), torch.tensor(y_tr))
    val_ds   = MultimodalASDDataset(torch.tensor(G_va), torch.tensor(B_va), torch.tensor(E_va), torch.tensor(y_va))
    test_ds  = MultimodalASDDataset(torch.tensor(G_te), torch.tensor(B_te), torch.tensor(E_te), torch.tensor(y_te))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, (G_te, B_te, E_te, y_te)
