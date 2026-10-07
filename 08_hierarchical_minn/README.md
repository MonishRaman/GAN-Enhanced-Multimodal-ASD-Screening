# Objective 3: Hierarchical Attention-Based Multi-Input Neural Network (MINN)

## Overview
This package implements **Objective 3** of the research roadmap:
> *"To develop a hierarchical attention-based multimodal feature fusion strategy integrated with a Multi-Input Neural Network for learning complementary genetic, environmental, and behavioral representations for ASD screening."*

---

## Data Flow & Objective Handoff Diagram

```mermaid
flowchart TD
    subgraph Obj1 ["Objective 1: Modality Representation Learning"]
        G_raw["Microarray Genetic Data"] --> G_trans["Genetic Transformer (128-D)"]
        B_raw["Clinical Behavioral Data"] --> B_dnn["Behavioral DNN (128-D)"]
        E_raw["Environmental Exposure Data"] --> E_dnn["Environmental DNN (128-D)"]
    end

    subgraph Obj2 ["Objective 2: Alignment & Adaptive GAN Augmentation"]
        G_trans --> G_align["128->32 Linear Alignment"]
        B_dnn --> B_align["128->32 Linear Alignment"]
        E_dnn --> E_align["128->32 Linear Alignment"]

        G_align --> Real_G["Real Genetic (92 Train: 69 ASD / 23 Control)"]
        Real_G --> GAN["Conditional GAN (Embedding Space)"]
        GAN --> Synth_G["+46 Synthetic Control Samples"]
        Real_G --> Balanced_G["Balanced Genetic Train Dataset\n(69 ASD / 69 Control = 138 samples)"]
        Synth_G --> Balanced_G
    end

    subgraph Obj3 ["Objective 3: Hierarchical Attention MINN"]
        Balanced_G --> G_branch["Genetic Branch (32->64->32)"]
        B_align --> B_branch["Behavioral Branch (32->64->32)"]
        E_align --> E_branch["Environmental Branch (32->64->32)"]

        G_branch --> Intra_G["Level 1: Intra-Modality Attention (α_G)"]
        B_branch --> Intra_B["Level 1: Intra-Modality Attention (α_B)"]
        E_branch --> Intra_E["Level 1: Intra-Modality Attention (α_E)"]

        Intra_G --> Inter_Att["Level 2: Dynamic Inter-Modality Attention (β_G, β_B, β_E)\n(Patient-level Softmax)"]
        Intra_B --> Inter_Att
        Intra_E --> Inter_Att

        Inter_Att --> Fusion["Adaptive Multimodal Fusion (96-D)"]
        Fusion --> Classifier["ASD Classifier (Dense + Dropout + Sigmoid)"]
        Classifier --> Out["ASD Prediction & Diagnostic Risk"]
    end

    subgraph Obj4 ["Objective 4: Clinical Explainability & Benchmarking"]
        Classifier --> Exp["Explainability Engine"]
        Intra_G --> Exp
        Inter_Att --> Exp
        Exp --> Clin["Feature Saliency, Dynamic Modality Breakdown & Counterfactuals"]
    end
```

---

## Directory Structure
```
08_hierarchical_minn/
├── data/                                 # Aligned 32-D and balanced handoff datasets
│   ├── genetic_32d_train_gan_balanced.csv
│   ├── genetic_32d_test_unseen.csv
│   ├── behavioral_32d_all.csv
│   └── environmental_32d_all.csv
├── models/
│   ├── hierarchical_minn.py              # PyTorch model definition
│   └── minn_best.pt                      # Saved best model checkpoint
├── results/
│   ├── minn_config.json                  # Model hyperparameter configuration
│   ├── training_history.csv              # Per-epoch training/val losses and AUCs
│   ├── model_benchmarking_comparison.csv # Comparison vs Unimodal & Multimodal Baselines
│   └── patient_level_attention_weights.csv
├── figures/
│   ├── training_curves.png
│   ├── roc_curve_benchmarking.png
│   ├── minn_confusion_matrix.png
│   ├── inter_modality_attention_distribution.png
│   └── intra_modality_feature_saliency.png
├── reports/
│   ├── objective3_minn_evaluation_report.txt
│   └── attention_mechanism_validation.txt
├── create_handoff_dataset.py             # Generates Objective 2 -> 3 handoff data
├── data_loader.py                        # Dataset & DataLoader utility
├── train_minn.py                         # Training pipeline with early stopping
├── evaluate_and_benchmark.py             # Evaluation and baseline benchmarking
└── inspect_attention.py                  # Dynamic attention validation
```

---

## How to Reproduce Each Stage

### 1. Generate the Objective 2 Balanced Handoff Dataset
```bash
python 08_hierarchical_minn/create_handoff_dataset.py
```
Projects GAN-augmented synthetic samples to 32-D space and outputs `genetic_32d_train_gan_balanced.csv`.

### 2. Train the Hierarchical Attention MINN
```bash
python 08_hierarchical_minn/train_minn.py
```
Trains the model with AdamW, Cosine Annealing, and saves `minn_best.pt` along with loss curves.

### 3. Evaluate and Benchmark Against Baselines
```bash
python 08_hierarchical_minn/evaluate_and_benchmark.py
```
Evaluates on the held-out unseen test set, generates confusion matrices and ROC curves, and compares against:
- Unimodal Genetic MLP
- Unimodal Behavioral MLP
- Unimodal Environmental MLP
- Multimodal Early Fusion MLP (No Attention)
- Calibrated Evidence Probability Fusion (M3.2 static weights)

### 4. Inspect Learned Attention Distributions
```bash
python 08_hierarchical_minn/inspect_attention.py
```
Extracts feature saliency ($\boldsymbol{\alpha}$) and patient-specific dynamic modality weights ($\beta$) to confirm dynamic variation.
