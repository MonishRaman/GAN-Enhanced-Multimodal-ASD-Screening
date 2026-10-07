"""
Objective 2 to Objective 3 Handoff Generator.

This script takes the verified Objective 2 GAN-generated synthetic samples,
projects them through the trained 128-to-32 alignment network, and constructs
the balanced multimodal dataset for Objective 3 training and evaluation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("d:/GAN-Enhanced-Multimodal-ASD-Screening")
GENETIC_MODEL_DIR = PROJECT_ROOT / "Genetic_Model"
BEHAVIORAL_MODEL_DIR = PROJECT_ROOT / "Behavioral_Model"
ENVIRONMENTAL_MODEL_DIR = PROJECT_ROOT / "Environmental_Model"
GAN_OUTPUT_DIR = PROJECT_ROOT / "Genetics_Complete_code" / "genetics" / "outputs" / "objective2_genetic"
HANDOFF_DIR = PROJECT_ROOT / "08_hierarchical_minn" / "data"

os.makedirs(HANDOFF_DIR, exist_ok=True)

# 1. Define Alignment Network Architecture exactly as in step38
class GeneticAlignmentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.network(x)

def generate_handoff_dataset():
    print("=" * 70)
    print("GENERATING OBJECTIVE 2 -> OBJECTIVE 3 BALANCED HANDOFF DATASET")
    print("=" * 70)

    # 1. Load trained Genetic 128-to-32 Alignment Network
    align_weights_path = GENETIC_MODEL_DIR / "04_cross_modal_alignment" / "genetic_128_to_32_alignment.pt"
    if not align_weights_path.exists():
        raise FileNotFoundError(f"Alignment weights not found at {align_weights_path}")
    
    align_model = GeneticAlignmentNetwork()
    align_model.load_state_dict(torch.load(align_weights_path, map_location="cpu"))
    align_model.eval()
    print(f"[OK] Loaded Genetic Alignment Network from {align_weights_path}")

    # 2. Load Real 32-D Aligned Representations
    genetic_real_32_df = pd.read_csv(GENETIC_MODEL_DIR / "04_cross_modal_alignment" / "genetic_32d_aligned_representation.csv")
    behavioral_32_df = pd.read_csv(BEHAVIORAL_MODEL_DIR / "04_cross_modal_alignment" / "behavioral_32d_aligned_representation.csv")
    environmental_32_df = pd.read_csv(ENVIRONMENTAL_MODEL_DIR / "04_cross_modal_alignment" / "environmental_32d_aligned_representation.csv")

    print(f"Real Genetic 32-D samples      : {len(genetic_real_32_df)} (ASD: {(genetic_real_32_df['Target'] == 1).sum()}, Control: {(genetic_real_32_df['Target'] == 0).sum()})")
    print(f"Real Behavioral 32-D samples   : {len(behavioral_32_df)} (ASD: {(behavioral_32_df['Class/ASD'] == 1).sum()}, Control: {(behavioral_32_df['Class/ASD'] == 0).sum()})")
    print(f"Real Environmental 32-D samples: {len(environmental_32_df)} (ASD: {(environmental_32_df['Target'] == 1).sum()}, Control: {(environmental_32_df['Target'] == 0).sum()})")

    # 3. Load Synthetic 128-D Control Embeddings from Objective 2 Embedding GAN
    synthetic_128_path = GAN_OUTPUT_DIR / "embedding_gan" / "synthetic_control_128d_embeddings.csv"
    if not synthetic_128_path.exists():
        raise FileNotFoundError(f"Synthetic embeddings not found at {synthetic_128_path}")
    
    synthetic_128_df = pd.read_csv(synthetic_128_path)
    print(f"[OK] Loaded {len(synthetic_128_df)} synthetic Control 128-D embeddings from Objective 2 GAN.")

    # Project synthetic 128-D embeddings to 32-D aligned space
    dim_cols = [c for c in synthetic_128_df.columns if c.startswith("Genetic_Embedding_")]
    assert len(dim_cols) == 128, f"Expected 128 features, got {len(dim_cols)}"
    
    synth_tensor = torch.tensor(synthetic_128_df[dim_cols].values, dtype=torch.float32)
    with torch.no_grad():
        synth_32_tensor = align_model(synth_tensor)
    
    synth_32_cols = [f"Genetic_32D_{i+1:03d}" for i in range(32)]
    synthetic_32_df = pd.DataFrame(synth_32_tensor.numpy(), columns=synth_32_cols)
    synthetic_32_df.insert(0, "Target", 0) # Control
    synthetic_32_df.insert(0, "Sample_Index", [f"SYNTH_CTRL_{i+1:03d}" for i in range(len(synthetic_32_df))])
    synthetic_32_df.insert(1, "Is_Synthetic", 1)

    # Standardize real genetic df columns
    real_g_df = genetic_real_32_df.copy()
    real_g_df.insert(1, "Is_Synthetic", 0)

    # Align column names
    g_32_cols_real = [c for c in real_g_df.columns if "32" in c or "dim" in c.lower() or c.startswith("Genetic_")]
    if len(g_32_cols_real) == 32:
        rename_map = {old: new for old, new in zip(g_32_cols_real, synth_32_cols)}
        real_g_df.rename(columns=rename_map, inplace=True)

    # Combine real training genetic samples and synthetic samples
    # Keep unseen test set isolated
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(
        np.arange(len(real_g_df)),
        test_size=24,
        random_state=SEED,
        stratify=real_g_df["Target"]
    )
    
    g_train_real = real_g_df.iloc[train_indices].copy()
    g_test_real = real_g_df.iloc[test_indices].copy()

    print(f"\nGenetic Real Split:")
    print(f"  Train: {len(g_train_real)} (ASD: {(g_train_real['Target'] == 1).sum()}, Control: {(g_train_real['Target'] == 0).sum()})")
    print(f"  Test : {len(g_test_real)} (ASD: {(g_test_real['Target'] == 1).sum()}, Control: {(g_test_real['Target'] == 0).sum()})")

    # Augment training set with synthetic Control samples
    g_train_balanced = pd.concat([g_train_real, synthetic_32_df], ignore_index=True)
    print(f"\nGenetic GAN-Augmented Training Set:")
    print(f"  Total samples: {len(g_train_balanced)}")
    print(f"  ASD samples  : {(g_train_balanced['Target'] == 1).sum()} (All Real)")
    print(f"  Ctrl samples : {(g_train_balanced['Target'] == 0).sum()} (Real: {(g_train_real['Target'] == 0).sum()}, Synthetic: {len(synthetic_32_df)})")
    print(f"  Balance ratio: {(g_train_balanced['Target'] == 1).sum()} : {(g_train_balanced['Target'] == 0).sum()} (1:1 Balanced)")

    # Save balanced datasets
    g_train_balanced.to_csv(HANDOFF_DIR / "genetic_32d_train_gan_balanced.csv", index=False)
    g_test_real.to_csv(HANDOFF_DIR / "genetic_32d_test_unseen.csv", index=False)
    synthetic_32_df.to_csv(HANDOFF_DIR / "synthetic_control_32d_aligned.csv", index=False)
    
    # Save Behavioral and Environmental 32-D
    behavioral_32_df.to_csv(HANDOFF_DIR / "behavioral_32d_all.csv", index=False)
    environmental_32_df.to_csv(HANDOFF_DIR / "environmental_32d_all.csv", index=False)

    print(f"\n[SUCCESS] Objective 2 -> Objective 3 Handoff files created in:\n  {HANDOFF_DIR}")

if __name__ == "__main__":
    generate_handoff_dataset()
