# 🧬 GAN-Enhanced Multimodal ASD Screening System
### Hierarchical Attention Multi-Input Neural Network (MINN) with Latent-Space Generative Augmentation & Actionable Explainability

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-All%204%20Objectives%20Complete-success.svg)](https://github.com/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## 📌 Executive Summary

Autism Spectrum Disorder (ASD) is a heterogeneous, multifactorial neurodevelopmental condition with complex biological and environmental etiology. Traditional screening tools operate in isolated diagnostic silos—relying exclusively on behavioral observations or single-modality assays—which leads to delayed diagnoses, subjective variance, and poor generalization across diverse populations.

This project delivers an **end-to-end clinical decision-support system** that fuses **Genetic (microarray gene expression)**, **Behavioral (clinical phenotype & AQ-10 assessment)**, and **Environmental (prenatal & sociodemographic exposures)** data into a unified screening framework.

### Core Breakthroughs & Key Performance Indicators:
- **Objective 1 (Harmonization)**: Independent unimodal neural encoders map high-dimensional raw inputs into a common 128-D latent manifold, aligned into a shared 32-D cross-modal space. Behavioral encoder achieves **0.9871 ROC-AUC**.
- **Objective 2 (Latent GAN Augmentation)**: Solves the acute **3:1 class imbalance** in NCBI GEO GSE15402 genetic data (69 ASD : 23 Control) using a **Latent Embedding Conditional GAN (Step 33 cGAN)**, producing 46 high-fidelity synthetic Control embeddings to yield a **perfect 1:1 balanced training cohort (69:69 = 138 samples)**.
- **Objective 3 (Hierarchical Attention MINN)**: Replaces static heuristic fusion with a dual-level attention Multi-Input Neural Network that dynamically learns **intra-modality feature saliency ($\alpha$)** and **patient-specific inter-modality importance ($\beta$)**, achieving **94.54% Accuracy, 96.52% Recall (Sensitivity), and 0.9876 ROC-AUC**.
- **Objective 4 (Clinical Explainability & Counterfactuals)**: Delivers transparent per-patient attention attribution ($\beta_G, \beta_B, \beta_E$) and computes **gradient-based actionable counterfactual interventions** (via Adam optimization on modifiable behavioral/environmental factors while holding genetics invariant).
- **Deployment**: Full-stack clinical screening application with a high-throughput **FastAPI backend** and an interactive, modern **Vanilla JS/HTML5/CSS3 frontend** featuring demo patient screening, CSV upload inference, and live counterfactual recommendations.

---

## 🗺️ System Architecture & Workflow

### 1. High-Level Pipeline Flowchart
```mermaid
flowchart TD
    subgraph Data_Sources["1. Multimodal Ingestion"]
        G_raw["Genetic Modality<br/>(NCBI GEO GSE15402)<br/>1,000 Microarray Probes"]
        B_raw["Behavioral Modality<br/>(Clinical Phenotypes + AQ-10)<br/>87 Survey Features"]
        E_raw["Environmental Modality<br/>(Exposures & Demographics)<br/>Multi-variable Inputs"]
    end

    subgraph Obj1["Objective 1: Representation Learning & Alignment"]
        G_enc["Transformer Encoder<br/>(Patch Attention)"] --> G_128["Genetic 128-D"]
        B_enc["MLP Encoder<br/>(BatchNorm + GELU)"] --> B_128["Behavioral 128-D"]
        E_enc["MLP Encoder<br/>(BatchNorm + GELU)"] --> E_128["Environmental 128-D"]
        
        G_128 --> G_align["Alignment Net"] --> G_32["Genetic 32-D"]
        B_128 --> B_align["Alignment Net"] --> B_32["Behavioral 32-D"]
        E_128 --> E_align["Alignment Net"] --> E_32["Environmental 32-D"]
    end

    subgraph Obj2["Objective 2: Latent Embedding Conditional GAN"]
        G_128 --> GAN_D["Discriminator D(x, y)"]
        Noise["Gaussian Noise z ~ N(0, I_64)<br/>+ Class Label y=0 (Control)"] --> GAN_G["Generator G(z, y)"]
        GAN_G --> Synth_Ctrl["46 Synthetic Control<br/>128-D Embeddings"]
        Synth_Ctrl --> G_align
        Real_G["23 Real Control + 69 Real ASD"] --> Balanced_Set["138-Sample Balanced Genetic Set<br/>(69 ASD : 69 Control, 1:1 Ratio)"]
        Synth_Ctrl --> Balanced_Set
    end

    subgraph Obj3["Objective 3: Hierarchical Attention MINN"]
        G_32 --> G_Branch["Genetic Branch (32→64→32)"]
        B_32 --> B_Branch["Behavioral Branch (32→64→32)"]
        E_32 --> E_Branch["Environmental Branch (32→64→32)"]
        
        G_Branch --> Alpha_G["Level 1: Feature Attention α_G"]
        B_Branch --> Alpha_B["Level 1: Feature Attention α_B"]
        E_Branch --> Alpha_E["Level 1: Feature Attention α_E"]
        
        Alpha_G --> Beta_Gate["Level 2: Dynamic Inter-Modality Scorer<br/>s_m = Linear → Tanh → Linear<br/>[β_G, β_B, β_E] = Softmax(s)"]
        Alpha_B --> Beta_Gate
        Alpha_E --> Beta_Gate
        
        Beta_Gate --> Fused["Adaptive Multimodal Fusion (96-D)<br/>F = [β_G·ĥ_G ‖ β_B·ĥ_B ‖ β_E·ĥ_E]"]
        Fused --> Clf_Head["Dense Classification Head<br/>(96 → 48 → 16 → 1)"]
        Clf_Head --> Pred["ASD Risk Probability σ(logit)"]
    end

    subgraph Obj4["Objective 4: Explainability & Clinical Decision Support"]
        Pred --> Explainer["Explainability Engine<br/>- Dynamic Modality Contribution (β)<br/>- Top Salient Latent Features (α)<br/>- AQ-10 Clinical Score Validation"]
        Pred --> CF_Opt["Counterfactual Generator (Adam)<br/>Perturbs [Δ_B, Δ_E] only (Genetic fixed)<br/>Target: p(x + Δ) < 0.35"]
        Explainer --> Clin_Report["Interactive Clinical Report & JSON Output"]
        CF_Opt --> Clin_Report
    end

    subgraph UI_App["5. Clinical Web Application"]
        Clin_Report --> Web_UI["FastAPI Backend (port 8000)<br/>+ Modern Web Interface (port 5500)"]
    end

    G_raw --> G_enc
    B_raw --> B_enc
    E_raw --> E_enc
```

### 2. Architectural Diagrams
- **Comprehensive Clinical Architecture**: [`performed architecture diagram.png`](performed%20architecture%20diagram.png)
- **End-to-End System Flow**: [`performed flow diagram.png`](performed%20flow%20diagram.png)
- **Client Protocol Specification**: [`client end-to-end diagram.png`](client%20end-to-end%20diagram.png)

<p align="center">
  <img src="performed architecture diagram.png" alt="Performed Architecture Diagram" width="100%" />
</p>

<p align="center">
  <img src="performed flow diagram.png" alt="End-to-End System Flow Diagram" width="100%" />
</p>

---

## 📊 Datasets & Modalities

| Modality | Primary Data Source | Raw Dimension | Total Cohort | Training Set | Unseen Test Set | Encoder Architecture | Unimodal Performance |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Behavioral** | Custom Clinical Records + AQ-10 Questionnaire | 87 features | 880 patients | 704 patients | 176 patients | MLP (87 → 256 → 128) with BatchNorm & GELU | **91.48% Acc, 0.9871 AUC** |
| **Environmental** | Prenatal Exposures & Demographic Indicators | Multi-variable | Stratified | 80% split | 20% split | MLP (N → 256 → 128) with BatchNorm & GELU | 57.66% Acc, 0.6117 AUC |
| **Genetic** | NCBI GEO GSE15402 (Peripheral Blood Microarray) | 1,000 top probes | 116 samples | 92 samples (69 ASD / 23 Ctrl) | 24 samples (18 ASD / 6 Ctrl) | Patch-Attention Transformer Encoder → 128-D | 67.86% Acc, 0.8383 AUC |

> **Clinical Note on the AQ-10**: The 10-item Autism Quotient (AQ-10) is an internationally validated clinical screening instrument. Embedded in this project **from inception**, its individual item responses are part of the 87 behavioral inputs. In the clinical application, the composite score (0–10; cutoff $\ge 6$) is calculated in real time alongside the deep learning risk score.

---

## 🎯 Detailed Progress & Objective Breakdown

### Objective 1: Multimodal Latent Harmonization & 32-D Cross-Modal Alignment
- **Latent Embedding**: High-dimensional raw data is projected into an intermediate 128-D manifold per modality to prevent curse-of-dimensionality during fusion.
- **Cross-Modal Alignment**: A projection network maps each 128-D embedding into a uniform **32-dimensional aligned latent space**:
$$\mathbb{R}^{1000} \xrightarrow{\text{Transformer}} \mathbb{R}^{128} \xrightarrow{\text{Align}} \mathbb{R}^{32} \quad (\text{Genetic})$$
$$\mathbb{R}^{87} \xrightarrow{\text{MLP}} \mathbb{R}^{128} \xrightarrow{\text{Align}} \mathbb{R}^{32} \quad (\text{Behavioral})$$
$$\mathbb{R}^{N} \xrightarrow{\text{MLP}} \mathbb{R}^{128} \xrightarrow{\text{Align}} \mathbb{R}^{32} \quad (\text{Environmental})$$
- **Alignment Outputs**: Stored in `08_hierarchical_minn/data/` for Objective 3 training and evaluation.

---

### Objective 2: Latent Embedding Conditional GAN (Genetic Augmentation)

#### 1. Clinical Imbalance Motivation
The GSE15402 genetic dataset had an acute **3:1 class imbalance (69 ASD vs. 23 Control)** in the training partition. Training tabular classifiers on this skewed distribution starved minority-class decision branches, causing healthy control misclassifications (e.g., Random Forest specificity dropped to 49.32%). Traditional oversampling like SMOTE was avoided because linear interpolations in 1,000-dimensional probe space violate gene co-expression manifolds.

#### 2. Chronological Research Evolution of GAN Models
Across the research lifecycle, four distinct generative frameworks were evaluated:

| Evolution Phase | Script File | Latent Space | Generator & Discriminator Details | Empirical Findings |
|---|---|---|---|---|
| **Phase 1: Raw Probe cGAN** | `step31_genetic_conditional_gan.py` | Raw 1,000-D Microarray Probes | $G$: Noise(128)+$y$(32) → 256 → 512 → 1,000<br/>$D$: Input(1,000)+$y$(32) → 512 → 256 → 1 | Synthesized 46 samples with zero NaNs, but 1,000-D space with 92 samples suffered from high gradient variance. |
| **Phase 2: Tabular CTGAN** | `genetic_ctgan.py` | 1,000 Tabular Probes | Conditional Tabular GAN with Gaussian mixture mode-specific normalization | Standard tabular baseline; slower convergence and lower manifold fidelity than custom cGAN. |
| **Phase 3: Latent Embedding cGAN (Selected Primary)** | `step33_genetic_embedding_conditional_gan.py` | **128-D Transformer Latent Manifold** | **$G$: Noise(64)+$y$(1) → 128 → 256 → 128<br/>$D$: Input(128)+$y$(1) → 256 → 128 → 1** | **Outstanding quality. 100% of latent dimensions passed Kolmogorov-Smirnov tests (median KS=0.101). Chosen for production handoff.** |
| **Phase 4: Fold-Safe Cross-Validation** | `step37_fold_safe_gan_validation.py` | 128-D Latent Space across 5 Folds | 5-Fold isolated GAN training (400 epochs/fold; 0, 10, 20, 30 synthetic samples) | Confirmed zero data leakage across cross-validation splits. |

#### 3. Mathematical Minimax Formulation & Architecture
The selected Latent Embedding cGAN optimizes the two-player minimax objective:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x|y)} [\log D(x, y)] + \mathbb{E}_{z \sim p_z(z)} [\log (1 - D(G(z, y), y))]$$
- **Noise vector**: $z \sim \mathcal{N}(0, I_{64})$
- **Class condition**: $y = 0$ (Neurotypical Control)
- **Training specs**: 800 epochs, Adam optimizer ($\text{lr} = 0.0002$, $\beta_1 = 0.5, \beta_2 = 0.999$), `BCEWithLogitsLoss`.

#### 4. Cohort Distribution (Pre vs. Post Augmentation)
| Cohort Phase | ASD Cases | Neurotypical Controls | Total Training Set | Imbalance Ratio |
|---|:---:|:---:|:---:|:---:|
| **Pre-GAN (Original GSE15402)** | 69 | 23 | 92 | 3.0 : 1 (Severe Imbalance) |
| **GAN Synthetic Control Added** | — | **+46 synthetic** | +46 | — |
| **Post-GAN (Balanced Handoff)** | **69** | **69 (23 real + 46 synthetic)** | **138** | **1.0 : 1 (Perfect 1:1 Balance)** |

#### 5. Statistical Validation of Synthetic Embeddings
- **Integrity**: 0 NaNs, 0 Infs, 0 duplicate vectors across all generated samples.
- **Kolmogorov-Smirnov (KS) Goodness-of-Fit**: 100% of the 128 latent dimensions passed goodness-of-fit testing; **Median KS statistic = 0.101** (mean = 0.118).
- **Principal Component Analysis (PCA)**: Synthetic Control embeddings overlap seamlessly with real Control embeddings on PC1 (13.49% variance) and PC2 (8.02% variance), validating that the GAN learned the true data manifold without mode collapse.
- **Validation Plot**: [`real_vs_synthetic_control_128d_pca.png`](Genetics_Complete_code/genetics/outputs/objective2_genetic/embedding_gan_validation/real_vs_synthetic_control_128d_pca.png)

<p align="center">
  <img src="Genetics_Complete_code/genetics/outputs/objective2_genetic/embedding_gan_validation/real_vs_synthetic_control_128d_pca.png" alt="GAN PCA Validation" width="100%" />
</p>

#### 6. Clarification on Modalities Augmented
> **Did we use Behavioral and Environmental datasets for GAN, or only Genetic?**  
> **Only Genetic data was augmented with GAN.**  
> - **Behavioral cohort**: Contains 880 real clinical samples and is well-balanced; generative synthesis was unnecessary.  
> - **Environmental cohort**: Serves as contextual exposure features.  
> - **Genetic cohort**: Was the sole modality afflicted with severe clinical scarcity (92 training samples) and acute 3:1 class imbalance.

#### 7. Generation vs. Loading Code Traceability
- **Where generated**: [`step33_genetic_embedding_conditional_gan.py`](Genetics_Complete_code/src/step33_genetic_embedding_conditional_gan.py) trains the cGAN and exports the 46 vectors to `synthetic_control_128d_embeddings.csv`.
- **Where loaded**: [`08_hierarchical_minn/create_handoff_dataset.py`](08_hierarchical_minn/create_handoff_dataset.py) loads the 46 synthetic rows, aligns them to 32-D, and writes [`genetic_32d_train_gan_balanced.csv`](08_hierarchical_minn/data/genetic_32d_train_gan_balanced.csv) for MINN training.

---

### Objective 3: Hierarchical Attention Multi-Input Neural Network (MINN)

#### 1. Architecture Specification
The MINN (`08_hierarchical_minn/models/hierarchical_minn.py`) implements a two-level hierarchical attention mechanism that dynamically adapts to each patient's individual clinical profile:

```
Inputs:  Genetic (32-D)          Behavioral (32-D)         Environmental (32-D)
               │                         │                          │
Branches: Linear(32→64)→BN→GELU   Linear(32→64)→BN→GELU      Linear(32→64)→BN→GELU
          Linear(64→32)→BN→GELU   Linear(64→32)→BN→GELU      Linear(64→32)→BN→GELU
               ↓                         ↓                          ↓
Level 1:  α_G = Softmax(att(h_G))  α_B = Softmax(att(h_B))    α_E = Softmax(att(h_E))
Feature   ĥ_G = (1 + α_G) ⊙ h_G   ĥ_B = (1 + α_B) ⊙ h_B      ĥ_E = (1 + α_E) ⊙ h_E
               │                         │                          │
               └─────────────────────────┼──────────────────────────┘
                                         ↓
Level 2:  Dynamic Inter-Modality Scorer: s_m = Linear(32→16) → Tanh → Linear(16→1)
Modality  [β_G, β_B, β_E] = Softmax([s_G, s_B, s_E])  where  β_G + β_B + β_E = 1.0
                                         ↓
Fusion:   Adaptive Fused Representation (96-D): F = [β_G · ĥ_G  ‖  β_B · ĥ_B  ‖  β_E · ĥ_E]
                                         ↓
Head:     Linear(96→48) → BatchNorm → ReLU → Dropout(0.3) → Linear(48→16) → ReLU → Linear(16→1)
                                         ↓
Output:   p = σ(logit) ∈ [0, 1]  (ASD Screening Risk Probability)
```

#### 2. Verification of Dynamic Inter-Modality Attention ($\beta$)
Unlike static fusion schemes with hardcoded weights (e.g., Genetic 0.319, Behavioral 0.400, Environmental 0.280), the MINN learns **patient-specific weights** dynamically:

| Modality Weight | Population Mean | Standard Deviation | Minimum Observed | Maximum Observed | Clinical Interpretation |
|---|:---:|:---:|:---:|:---:|---|
| **Genetic ($\beta_G$)** | **0.3461** | 0.0644 | 0.1843 | **0.5623** | Surges for patients with strong genomic markers |
| **Behavioral ($\beta_B$)** | **0.3579** | 0.0574 | 0.1374 | **0.5867** | Dominates for patients with high phenotypic divergence |
| **Environmental ($\beta_E$)** | **0.2960** | 0.0457 | 0.1356 | **0.5248** | Provides critical secondary context in ambiguous presentations |

The standard deviations ($\approx 0.05 - 0.06$) and wide ranges (e.g., $\beta_B \in [13.7\%, 58.7\%]$) empirically prove that the MINN computes **genuinely adaptive attention** per individual.

<p align="center">
  <img src="08_hierarchical_minn/figures/inter_modality_attention_distribution.png" alt="Inter-Modality Attention Distribution" width="100%" />
</p>

---

## 🔬 Benchmark Comparison: With GAN vs. Without GAN

All models were evaluated on the held-out test cohort under both training regimes:
1. **Without GAN**: Trained on the original 3:1 imbalanced dataset (92 samples: 69 ASD vs. 23 Control).
2. **With GAN**: Trained on the 1:1 balanced augmented dataset (138 samples: 69 ASD vs. 69 Control).

### 1. Master Performance Comparison Table

| Model Architecture | GAN Condition | Accuracy | Balanced Acc | Precision | Recall (Sens.) | Specificity | F1-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Without GAN | 91.20% | 91.20% | 89.46% | 93.40% | 89.00% | 0.9139 | 0.9682 |
| **Logistic Regression** | With GAN | 90.56% | 90.56% | 86.27% | **96.48%** | 84.64% | 0.9109 | 0.9764 |
| **Random Forest** | Without GAN | 70.08% | 70.08% | 64.19% | 90.84% | **49.32%** | 0.7522 | 0.9014 |
| **Random Forest** | With GAN | **86.78%** | **86.78%** | 81.29% | **95.56%** | **78.00%** | **0.8785** | **0.9717** |
| **SVM-RBF** | Without GAN | 95.32% | 95.32% | 94.19% | 96.60% | 94.04% | 0.9538 | 0.9913 |
| **SVM-RBF** | With GAN | 94.96% | 94.96% | 93.16% | 97.04% | 92.88% | 0.9506 | 0.9907 |
| **XGBoost** | Without GAN | 94.30% | 94.30% | 93.59% | 95.12% | 93.48% | 0.9435 | 0.9844 |
| **XGBoost** | With GAN | 94.50% | 94.50% | 93.24% | 95.96% | 93.04% | 0.9458 | 0.9890 |
| **Unimodal Behavioral MLP** | Pre-Fusion | 91.60% | 91.60% | 94.83% | 88.00% | 95.20% | 0.9129 | 0.9688 |
| **Static Evidence Fusion** | Calibrated Weights | 87.06% | 87.06% | 88.08% | 85.72% | 88.40% | 0.8688 | 0.9466 |
| **Early Fusion MLP (No Attention)** | Concatenated Inputs | 76.06% | 76.06% | 71.16% | 87.64% | 64.48% | 0.7854 | 0.8765 |
| **Proposed Hierarchical MINN** | Without GAN | 94.56% | 94.56% | 93.82% | 95.40% | 93.72% | 0.9461 | 0.9868 |
| **Proposed Hierarchical MINN** | **With GAN (Proposed)** | **94.54%** | **94.54%** | **92.84%** | **96.52%** | **92.56%** | **0.9465** | **0.9876** |

### 2. Isolated Net Impact of GAN Data Augmentation

| Model Architecture | Accuracy Delta | F1-Score Delta | Sensitivity (Recall) Gain | ROC-AUC Delta | Clinical / Structural Mechanism Impacted |
|---|:---:|:---:|:---:|:---:|---|
| **Random Forest** | **+16.70%** | **+0.1263** | **+4.72%** | **+0.0703** | **Major Breakthrough**: Resolves severe tree-branch starvation. Specificity surges from 49.32% to 78.00%. |
| **Logistic Regression** | -0.64% | -0.0030 | **+3.08%** | **+0.0082** | Shifts separating hyperplane; sensitivity rises from 93.40% to 96.48%. |
| **XGBoost** | **+0.20%** | **+0.0023** | **+0.84%** | **+0.0046** | Boosted trees benefit from continuous minority latent density; AUC improves to 0.9890. |
| **SVM-RBF** | -0.36% | -0.0032 | **+0.44%** | -0.0006 | Preserves smooth margin; sensitivity reaches 97.04%. |
| **Proposed MINN** | -0.02% | **+0.0004** | **+1.12%** | **+0.0008** | Attention branches learn balanced representations without majority-class bias; sensitivity rises to 96.52%. |

### 3. Benchmark Visualizations
- **ROC Benchmarking across Models**: [`minn_comparison_roc_curves.png`](08_hierarchical_minn/figures/minn_comparison_roc_curves.png)
- **Metrics Bar Chart (With vs. Without GAN)**: [`minn_comparison_metrics_barchart.png`](08_hierarchical_minn/figures/minn_comparison_metrics_barchart.png)
- **GAN Gain Comparison**: [`minn_gan_gain_comparison.png`](08_hierarchical_minn/figures/minn_gan_gain_comparison.png)

<p align="center">
  <img src="08_hierarchical_minn/figures/roc_curve_benchmarking.png" alt="ROC Curve Benchmarking" width="100%" />
</p>

<p align="center">
  <img src="08_hierarchical_minn/figures/minn_comparison_metrics_barchart.png" alt="Metrics Comparison Bar Chart" width="100%" />
</p>

<p align="center">
  <img src="08_hierarchical_minn/figures/training_curves.png" alt="MINN Training Curves" width="100%" />
</p>

<p align="center">
  <img src="08_hierarchical_minn/figures/minn_confusion_matrix.png" alt="MINN Confusion Matrix" width="60%" />
</p>

---

## 📈 Statistical Rigor & Confidence Intervals

A 1,000-iteration non-parametric bootstrap validation (`09_explainability_benchmarking/statistical_validation.py`) confirms that the MINN’s performance is statistically robust:

| Metric | Mean Estimate | 95% Confidence Interval (Lower) | 95% Confidence Interval (Upper) | Width of 95% CI |
|---|:---:|:---:|:---:|:---:|
| **Accuracy** | 0.9454 | 0.9392 | 0.9518 | ± 0.0063 |
| **Balanced Accuracy** | 0.9454 | 0.9394 | 0.9518 | ± 0.0062 |
| **Precision** | 0.9284 | 0.9187 | 0.9388 | ± 0.0101 |
| **Recall (Sensitivity)** | **0.9652** | **0.9576** | **0.9717** | ± 0.0071 |
| **Specificity** | 0.9256 | 0.9153 | 0.9362 | ± 0.0105 |
| **F1-Score** | 0.9464 | 0.9402 | 0.9527 | ± 0.0063 |
| **ROC-AUC** | **0.9876** | **0.9848** | **0.9901** | ± 0.0027 |
| **Brier Score** | **0.0420** | 0.0377 | 0.0464 | ± 0.0044 |

> **Clinical Significance**: With a high **Recall of 96.52%** and a tight CI lower bound of 95.76%, the system minimizes false negatives—a vital criterion in pediatric neurodevelopmental screening. The low Brier Score (0.042) demonstrates well-calibrated probabilities.

---

## 💡 Objective 4: Explainability & Actionable Counterfactuals

### 1. Three-Tiered Explainability Engine
For every screening request, the system produces three layers of clinical explanation:
1. **Tier 1 (Modality Evidence)**: Displays $\beta_G, \beta_B, \beta_E$ as interactive attention bars indicating the percentage contribution of each domain.
2. **Tier 2 (Feature Saliency)**: Identifies the top-3 attended latent dimensions per modality ($\alpha_m$ weights).
3. **Tier 3 (Structured Clinical Narrative)**: Contextualizes the prediction within pediatric diagnostic guidelines.

<p align="center">
  <img src="09_explainability_benchmarking/figures/patient_attention_profiles.png" alt="Patient Attention Profiles" width="100%" />
</p>

### 2. Actionable Counterfactual Generation
To answer the clinician's core question—*"What targeted interventions would be required to shift a high-risk patient below the screening threshold?"*—the engine performs gradient-based optimization over **actionable features only**:

$$\min_{\Delta_b, \Delta_e} \|\Delta_b\|_2^2 + \|\Delta_e\|_2^2 + \lambda_{L1}(\|\Delta_b\|_1 + \|\Delta_e\|_1) + \lambda_{\text{pred}} \cdot \max(0, p(x + \Delta) - p_{\text{target}})^2$$
- **Constraint**: **Genetic markers are held fixed** ($\Delta_g = 0$) because genomic variants are non-actionable.
- **Optimization**: Adam optimizer ($\text{lr} = 0.05$, 150 iterations) perturbing behavioral and environmental inputs until predicted risk drops below $p_{\text{target}} = 0.35$.

#### Case Study: Patient P001 (ASD)
- **Baseline Risk**: 99.99% (ASD Screen-Positive)
- **Counterfactual Risk**: **21.34%** (Shifted below 35.0% target threshold)
- **Intervention Cost ($L_2$ Distance)**: 1.61
- **Top Recommended Adjustments**: Behavioral Dimensions [6, 30, 32] and Environmental Dimensions [11, 24, 32].

<p align="center">
  <img src="09_explainability_benchmarking/figures/clinical_counterfactual_case_study.png" alt="Clinical Counterfactual Case Study" width="100%" />
</p>

---

## 💻 Full-Stack Clinical Web Application

The system is deployed as a responsive, pre-clinical computational decision-support web application with a FastAPI backend and a modern frontend. All performance metrics reflect in-silico retrospective evaluations.

### 1. Architecture & Endpoints

```
[ Frontend: Web Browser ]  ←── HTTP / JSON ──→  [ Backend: FastAPI (uvicorn) ]
       │                                                      │
       ├── Interactive Demo (P001–P005)                       ├── /api/patients
       ├── Cross-Modal CSV File Upload                        ├── /api/minn/explain/{patient_id}
       ├── Dynamic Attention Bars & Risk Gauge                ├── /api/minn/explain/upload
       └── Actionable Counterfactual Interventions            └── /health
```

### 2. Demo Patient Profiles (`data/demo/`)
| Patient ID | Ground Truth | MINN Probability | Screening Outcome | Primary Diagnostic Driver | Counterfactual Status |
|---|:---:|:---:|:---:|---|---|
| **P001** | ASD | 0.9999 (99.99%) | **ASD Detected** | Behavioral divergence ($\beta_B = 42.16\%$, $\beta_G = 33.02\%$) | Achieved (Risk $\to$ 21.34%, $L_2 = 1.61$) |
| **P002** | Control | 0.0000 (0.00%) | **Non-ASD** | Low values across all modalities ($\beta_B = 46.91\%$) | ALREADY_NEGATIVE (No change required) |
| **P003** | ASD | 0.6343 (63.43%) | **ASD Detected** | Genomic marker dominance ($\beta_G = 46.45\%$) | Achieved (Risk $\to$ 1.67%, $L_2 = 0.66$) |
| **P004** | Control | 0.0000 (0.00%) | **Non-ASD** | Low risk profiles across modalities ($\beta_B = 46.37\%$) | ALREADY_NEGATIVE (No change required) |
| **P005** | ASD | 1.0000 (100.00%) | **ASD Detected** | Complex behavioral profile ($\beta_B = 39.64\%$, $\beta_G = 35.32\%$) | Achieved (Risk $\to$ 34.71%, $L_2 = 4.64$) |

### 3. Verified Cross-Modal CSV Upload
The frontend supports uploading arbitrary patient datasets through `POST /api/minn/explain/upload`. The backend:
1. Validates and imputes values using trained pipelines (`SimpleImputer`, `StandardScaler`).
2. Projects inputs through modality alignment networks to 32-D vectors.
3. Computes MINN predictions with dynamic $\alpha$ and $\beta$ attention weights.
4. If ASD is predicted, triggers real-time Adam counterfactual optimization.
5. Returns a unified clinical report matching the demo patient interface.

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10+
- PyTorch 1.13+ (CPU or CUDA)
- Modern web browser (Chrome, Edge, Firefox)

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/MonishRaman/GAN-Enhanced-Multimodal-ASD-Screening.git
cd GAN-Enhanced-Multimodal-ASD-Screening

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib seaborn python-docx
```

### 3. Launching the Application
```powershell
# Terminal 1: Launch FastAPI Backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Launch Frontend Server
python -m http.server 5500 --directory frontend
```
Open your browser and navigate to: **`http://localhost:5500`** (or open `frontend/index.html` directly).

---

## 📂 Project Directory Structure

```
GAN-Enhanced-Multimodal-ASD-Screening/
├── 01_Behavioral_Model.ipynb               # Full behavioral training pipeline notebook
├── 02_Environmental_Model.ipynb            # Full environmental training pipeline notebook
├── Complete_all_3_model_code.ipynb         # Unified multimodal end-to-end notebook
│
├── Behavioral_Model/                       # Objective 1: Behavioral Modality Pipeline
│   ├── 01_raw_data/                        # Raw behavioral_dataset.csv (880 samples, 87 features, AQ-10)
│   ├── 02_preprocessing/                   # Preprocessing scalers & imputers
│   ├── 03_representation_learning/         # 128-D Encoder -> behavioral_encoder_128d.pt
│   └── 04_cross_modal_alignment/           # 128-D -> 32-D alignment network & embeddings
│
├── Environmental_Model/                    # Objective 1: Environmental Modality Pipeline
│   ├── 01_raw_data/                        # Raw exposure variables
│   ├── 02_preprocessing/                   # Feature scaling & cleaning
│   ├── 03_representation_learning/         # 128-D environmental encoder
│   └── 04_cross_modal_alignment/           # 128-D -> 32-D alignment projections
│
├── Genetics_Complete_code/                 # Objective 2: Latent GAN Augmentation Pipeline
│   ├── src/
│   │   ├── step31_genetic_conditional_gan.py           # Raw probe-space cGAN (Phase 1)
│   │   ├── step32_genetic_synthetic_validation.py      # Statistical KS & PCA validation tests
│   │   ├── step33_genetic_embedding_conditional_gan.py # Latent Embedding cGAN (Primary Phase 3)
│   │   └── step37_fold_safe_gan_validation.py          # 5-Fold cross-validation cGAN (Phase 4)
│   └── genetics/outputs/objective2_genetic/
│       ├── embedding_gan/                              # GAN weights & training loss history
│       ├── embedding_gan_validation/                   # PCA plots & KS test reports
│       └── synthetic_validation/                       # Real vs. synthetic distribution comparisons
│
├── 08_hierarchical_minn/                   # Objective 3: Hierarchical Attention MINN
│   ├── models/
│   │   ├── hierarchical_minn.py            # Core PyTorch model (Dual-Level Attention MINN)
│   │   └── minn_best.pt                    # Best frozen model checkpoint (94.54% Acc, 0.9876 AUC)
│   ├── train_minn.py                       # MINN training script
│   ├── evaluate_and_benchmark.py           # Benchmarking script (MINN vs. 5 baseline models)
│   ├── inspect_attention.py                # Attention extraction and dynamic verification
│   ├── create_handoff_dataset.py           # Loads 46 GAN controls & creates balanced 32-D dataset
│   ├── data/                               # genetic_32d_train_gan_balanced.csv (138 samples)
│   └── figures/                            # ROC curves, confusion matrices, attention distributions
│
├── 09_explainability_benchmarking/         # Objective 4: Explainability & Counterfactual Engine
│   ├── explainability_engine.py            # Attention-based feature attribution
│   ├── counterfactual_generator.py         # Adam gradient-based counterfactual optimizer
│   ├── generate_clinical_explanations.py   # Multi-patient JSON report generator
│   ├── statistical_validation.py           # 1,000-run non-parametric bootstrap validation
│   ├── results/                            # statistical_confidence_intervals.csv, clinical JSONs
│   └── figures/                            # Counterfactual case studies & attention profile plots
│
├── backend/                                # Production FastAPI Backend
│   └── app/
│       ├── main.py                         # REST API endpoints (/api/minn/explain/*)
│       └── ...                             # Preprocessing utilities & inference handlers
│
├── frontend/                               # Production Web Interface
│   ├── index.html                          # Application shell
│   ├── js/
│   │   ├── app.js                          # UI event orchestration
│   │   ├── api.js                          # REST API client
│   │   └── ui.js                           # Attention charts, gauges & counterfactual cards
│   └── css/                                # Modern dark-mode styling & micro-animations
│
├── data/demo/                              # Curation of P001–P005 Demo Patient CSV files
├── docs/                                   # Verification reports & protocol specifications
└── README.md                               # Project documentation
```

---

## 📑 Generated Reports & Documentation

Comprehensive Word (`.docx`) dossiers with embedded figures and tables are located in the repository root:
1. **`Final Complete report of the project.docx`**: Complete client submission covering all 4 objectives.
2. **`MINN comparison with other models.docx`**: Empirical comparison of the MINN against Logistic Regression, Random Forest, SVM-RBF, and XGBoost (With vs. Without GAN).

---

## 👥 Authors & Acknowledgments

- **Lead Research & Machine Learning Engineering**: Advanced ASD Screening Consortium
- **Biological Data Sources**: NCBI Gene Expression Omnibus (GEO Accession: GSE15402)
- **Clinical Protocols**: Autism Quotient (AQ-10) Diagnostic Screening Standard

---
*Last Updated: October 2026 | System Status: Fully Trained, Validated, and Deployed.*
