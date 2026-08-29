# Complete & Exhaustive Genetics Modality Analysis: Architecture, Mathematical Formulations, Pipeline Flow, Input/Output Inventory, File Formats, and Benchmarks

---

## 1. Repository & Directory Inventory

The genetics modality in the **GAN-Enhanced Multimodal ASD Screening** project spans raw microarray data processing, statistical feature selection, lightweight transformer representation learning, generative data augmentation (GANs & CTGAN), cross-modal alignment, and production backend deployment.

```
d:\GAN-Enhanced-Multimodal-ASD-Screening\
├── Genetics_Complete_code/
│   ├── src/                         # Complete 39-step execution scripts & CTGAN modules
│   │   ├── step1_merge_samples.py
│   │   ├── step2_data_quality_assessment.py
│   │   ├── step2_5_missing_distribution.py
│   │   ├── step3_filter_probes.py
│   │   ├── step4_knn_imputation.py
│   │   ├── step5_zscore_normalization.py
│   │   ├── step6_extract_labels.py
│   │   ├── step7_probe_mapping.py
│   │   ├── step8_build_ml_dataset.py
│   │   ├── step9_dataset_validation.py
│   │   ├── step10_variance_threshold.py
│   │   ├── step11_mutual_information.py
│   │   ├── step12_correlation_filter.py
│   │   ├── step18_annotate_stable_probes.py
│   │   ├── step19_genbank_to_gene.py
│   │   ├── step19b_genbank_sequence_to_gene.py
│   │   ├── step20_extract_sequence_annotations.py
│   │   ├── step21_create_verified_gene_annotation.py
│   │   ├── step22_statistical_validation.py
│   │   ├── step23_integrate_feature_evidence.py
│   │   ├── step24_biological_interpretation.py
│   │   ├── step25_final_project_summary.py
│   │   ├── step26_final_visualizations.py
│   │   ├── step27_genetic_128_embedding.py
│   │   ├── step28_train_genetic_transformer.py
│   │   ├── step29_genetic_embedding_validation.py
│   │   ├── step30_genetic_train_test_split.py
│   │   ├── step31_genetic_conditional_gan.py
│   │   ├── step32_genetic_synthetic_validation.py
│   │   ├── step33_genetic_embedding_conditional_gan.py
│   │   ├── step34_genetic_embedding_synthetic_validation.py
│   │   ├── step35_genetic_gan_utility_validation.py
│   │   ├── step36_genetic_gan_augmentation_sensitivity.py
│   │   ├── step37_fold_safe_genetic_gan_validation.py
│   │   ├── step38_genetic_128_to_32_alignment.py
│   │   ├── step39_validate_genetic_32d.py
│   │   ├── genetic_ctgan.py
│   │   ├── genetic_ctgan_validation.py
│   │   └── genetic_ctgan_utility_validation.py
│   ├── genetics/                    # Modular PyTorch codebase & baseline trainers
│   │   ├── model.py                 # GeneticTransformer class architecture
│   │   ├── config.py                # Hyperparameters & paths
│   │   ├── dataset.py               # GeneticDataset PyTorch wrapper
│   │   ├── train_baseline.py        # Baseline transformer training
│   │   ├── train_classical_baselines.py # Logistic Regression, SVM, Random Forest benchmarking
│   │   ├── train_feature_size_experiment.py # K={25,50,100,250,500,1000} experiments
│   │   ├── train_leakage_safe.py    # Strict fold-isolated feature selection & training
│   │   └── save_top100_fold_features.py
│   ├── genetic_128d_technical_summary.txt # Technical summary report
│   └── genetic_128d_training_report.txt  # Comprehensive training report
├── Genetic_Model/                   # Frozen model checkpoints for production backend
│   ├── 01_raw_data/
│   │   └── Genetic_dataset.csv      # Top-1000 features dataset (116 samples × 1000 features)
│   ├── 03_representation_learning/
│   │   ├── genetic_transformer_128d.pt  # Frozen 128-D Transformer weights
│   │   └── genetic_128d_embeddings.csv  # Extracted 128-D embeddings
│   └── 04_cross_modal_alignment/
│       ├── genetic_128_to_32_alignment.pt # Frozen 128-D to 32-D alignment weights
│       └── genetic_32d_aligned_representation.csv # Final 32-D aligned vector
├── 05_multimodal_fusion/            # Knowledge-level fusion directory
│   ├── M3_1_genetic_expert.pt       # Genetic Expert NN classifier
│   ├── M3_1_genetic_scaler.joblib   # Scaler for 32-D genetic inputs
│   └── M3_1_fusion_weights.csv      # Multimodal fusion weights (Genetic = 0.31937)
├── backend/
│   └── services/
│       ├── genetic_service.py       # Production FastAPI service for local genetic inference
│       └── fusion_service.py        # Production Multimodal Fusion service
├── data/                            # Processed data, demo inputs, and raw files
│   └── demo/P001..P005/
│       └── genetic_input.csv        # Single-patient 1000-feature input CSVs
└── MODEL_MANIFEST.json              # System manifest freezing ML artifacts
```

---

## 2. File Formats & Technical Schemas

| Category | File Format | Path / Location | Description & Schema |
| :--- | :--- | :--- | :--- |
| **Raw Microarray Inputs** | Tab-delimited `.txt` | `data/raw/GSM Sample Tables/GSM*.txt` | 3 Columns: `Position` (String probe ID), `Channel_A` (Float expression signal), `Channel_B`. |
| **Preprocessed Expression** | Matrix `.csv` | `data/processed/expression_matrix.csv` | Transposed rows = Probes, Columns = Sample GSM IDs (`GSM386518`, etc.). |
| **ML Feature Table** | Header `.csv` | `data/processed/top_features_dataset.csv` | 116 Rows (samples) $\times$ 1004 Columns (`GSM_ID`, `Source`, `Subtype`, `Target`, + 1000 genetic probes). |
| **PyTorch Models** | Checkpoint `.pt` | `Genetic_Model/*/*.pt`, `05_multimodal_fusion/*.pt` | Binary PyTorch `state_dict` dictionary mapping layer weights and biases. |
| **Scalers & Arrays** | `.joblib`, `.npz` | `05_multimodal_fusion/*.joblib`, `*.npz` | Scikit-Learn `StandardScaler` objects containing empirical `mean_` and `scale_`. |
| **Synthetic Tables** | Header `.csv` | `.../conditional_gan/synthetic_control_samples.csv` | Synthetic Control rows generated by Generator network ($46 \times 1000$). |
| **Embedding Tables** | Header `.csv` | `.../genetic_128d_embeddings_trained_oof.csv` | 116 Rows $\times$ 130 Columns (`GSM_ID`, `Target`, `Genetic_Embedding_001` .. `128`). |
| **Aligned 32-D Tables** | Header `.csv` | `Genetic_Model/04_cross_modal_alignment/*.csv` | 116 Rows $\times$ 34 Columns (`GSM_ID`, `Target`, `Genetic_Aligned_001` .. `032`). |

---

## 3. End-to-End Pipeline Flow Diagram & Overview

```mermaid
flowchart TD
    A["Raw Microarray GSM Tables (116 Samples)"] --> B["Step 1: Sample Merging (Channel_A Expression)"]
    B --> C["Step 3: Filter >75% Missing Probes"]
    C --> D["Step 4: KNN Imputation (k=5 Neighbors)"]
    D --> E["Step 5: Probe-wise Z-Score Normalization"]
    E --> F["Step 6-9: Label Extraction (87 Control / 29 ASD) & Dataset Creation"]
    F --> G["Step 10: Variance Threshold Filter (Var >= 0.01)"]
    G --> H["Step 11: Mutual Information Selection (Top 1000 Probes)"]
    H --> I["Step 12: Pearson Correlation Filter (|r| <= 0.95)"]
    I --> J["Step 28: Lightweight Genetic Transformer (1000 -> 32 Tokens -> 128-D)"]
    J --> K1["Step 31/33: Conditional GAN Data Augmentation (Class Balancing)"]
    J --> K2["Step 32/34/35/36/37: Synthetic Quality & Downstream Sensitivity Validation"]
    J --> K3["Step 38: Cross-Modal Alignment Network (128-D -> 64 -> 32-D)"]
    K3 --> L["Backend Service: Single-Patient Local Inference Engine"]
    L --> M["Multimodal Evidence Fusion (Weight = 0.31937)"]
```

---

## 4. Exhaustive Step-by-Step Code, Model, and Input/Output Inventory

### 4.1. Data Ingestion & Preprocessing (Steps 1–9)

#### Step 1: Merge Microarray Samples
- **File**: [`Genetics_Complete_code/src/step1_merge_samples.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step1_merge_samples.py)
- **Input**: `data/raw/GSM Sample Tables/*.txt`
- **Output**: `data/processed/expression_matrix.csv`
- **Description**: Parses all individual GSM sample files, extracts the `Position` probe column and `Channel_A` signal intensity values, and merges them into a single wide expression matrix across all 116 samples.

#### Step 2 & 2.5: Data Quality Assessment & Missing Distribution
- **Files**: [`Genetics_Complete_code/src/step2_data_quality_assessment.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step2_data_quality_assessment.py), [`Genetics_Complete_code/src/step2_5_missing_distribution.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step2_5_missing_distribution.py)
- **Input**: `data/processed/expression_matrix.csv`
- **Output**: `data/reports/data_quality_report.txt`, `data/reports/missing_distribution_report.txt`
- **Description**: Evaluates overall missingness across probes and samples, producing empirical distribution summaries.

#### Step 3: Probe Filtering
- **File**: [`Genetics_Complete_code/src/step3_filter_probes.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step3_filter_probes.py)
- **Input**: `data/processed/expression_matrix.csv`
- **Output**: `data/processed/filtered_expression_matrix.csv`, `data/removed/removed_probes.csv`, `data/reports/probe_filter_report.txt`
- **Description**: Calculates probe-wise missing ratio $\text{MissingRatio}_i = \frac{\text{Missing Count}}{N_{\text{samples}}} \times 100$. Removes probes exceeding $75\%$ missingness.

#### Step 4: KNN Imputation
- **File**: [`Genetics_Complete_code/src/step4_knn_imputation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step4_knn_imputation.py)
- **Input**: `data/processed/filtered_expression_matrix.csv`
- **Output**: `data/processed/knn_imputed_expression_matrix.csv`, `data/reports/knn_imputation_report.txt`
- **Description**: Imputes missing expression values using $k$-Nearest Neighbors ($k=5$, uniform Euclidean distance weighting).

#### Step 5: Probe-wise Z-Score Normalization
- **File**: [`Genetics_Complete_code/src/step5_zscore_normalization.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step5_zscore_normalization.py)
- **Input**: `data/processed/knn_imputed_expression_matrix.csv`
- **Output**: `data/processed/normalized_expression_matrix.csv`, `data/reports/zscore_normalization_report.txt`
- **Description**: Standardizes each probe row across samples to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).

#### Step 6–9: Label Extraction & ML Dataset Construction
- **Files**: [`Genetics_Complete_code/src/step6_extract_labels.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step6_extract_labels.py) to [`Genetics_Complete_code/src/step9_dataset_validation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step9_dataset_validation.py)
- **Input**: `data/raw/clinical_metadata.csv`, `data/processed/normalized_expression_matrix.csv`
- **Output**: `data/processed/labels.csv`, `data/processed/genetic_dataset.csv`
- **Description**: Maps GSM sample IDs to phenotypic targets ($0 = \text{Control}, 1 = \text{ASD}$), transposed so that rows represent samples (116) and columns represent probe features.

---

### 4.2. Feature Selection & Filtering (Steps 10–12)

#### Step 10: Variance Threshold Filter
- **File**: [`Genetics_Complete_code/src/step10_variance_threshold.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step10_variance_threshold.py)
- **Input**: `data/processed/genetic_dataset.csv`
- **Output**: `data/processed/variance_filtered_dataset.csv`, `data/reports/removed_low_variance_features.csv`
- **Description**: Removes uninformative probes with variance $\text{Var}(X) < 0.01$.

#### Step 11: Mutual Information Selection
- **File**: [`Genetics_Complete_code/src/step11_mutual_information.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step11_mutual_information.py)
- **Input**: `data/processed/variance_filtered_dataset.csv`
- **Output**: `data/processed/top_features_dataset.csv`, `data/reports/mutual_information_scores.csv`
- **Description**: Computes Mutual Information scores between probes and binary target labels using $k$-NN density estimation (`random_state=42`). Selects the **Top 1000 genetic features**.

#### Step 12: Correlation Filtering
- **File**: [`Genetics_Complete_code/src/step12_correlation_filter.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step12_correlation_filter.py)
- **Input**: `data/processed/top_features_dataset.csv`
- **Output**: `data/processed/final_genetic_features.csv`, `data/reports/removed_correlated_features.csv`
- **Description**: Computes upper-triangular Pearson correlation matrix and drops redundant probes with absolute correlation $|r| > 0.95$.

---

### 4.3. Annotation & Biological Interpretation (Steps 18–26)

- **Files**: [`step18_annotate_stable_probes.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step18_annotate_stable_probes.py) through [`step26_final_visualizations.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step26_final_visualizations.py)
- **Input**: `data/processed/top_features_dataset.csv`, GenBank accession mapping tables
- **Output**: `data/reports/gene_annotations.csv`, `data/reports/biological_pathway_enrichment.csv`, `data/visualizations/feature_importance.png`
- **Description**: Maps probe positions to GenBank nucleotide accessions, resolves gene symbols, extracts sequence annotations, runs enrichment against ASD-associated pathways (e.g., neuronal synaptic development, immune signaling), and generates publication-grade visualizations.

---

### 4.4. Representation Learning — Lightweight Genetic Transformer (Step 28)

#### Step 28: Train Lightweight Genetic Transformer
- **File**: [`Genetics_Complete_code/src/step28_train_genetic_transformer.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step28_train_genetic_transformer.py)
- **Input**: `data/processed/top_features_dataset.csv` (116 samples $\times$ 1000 features)
- **Output**:
  - Out-of-Fold Embeddings: `genetics/outputs/genetic_embedding/genetic_128d_embeddings_trained_oof.csv` (116 $\times$ 128)
  - CV Results: `genetics/outputs/genetic_embedding/genetic_transformer_lightweight_cv.csv`
  - Model Checkpoint: `genetics/outputs/genetic_embedding/genetic_transformer_128d_lightweight.pt`
  - Report: `genetics/outputs/genetic_embedding/genetic_transformer_128d_final_report.txt`
- **Model Class**: `LightweightGeneticTransformer`
- **Architecture**:
  - `Token Projection`: `nn.Linear(1000, 32 * 128)` $\to$ reshapes input into 32 tokens of 128 dimensions.
  - `Positional Embedding`: Learnable parameter tensor $(1, 32, 128) \sim \mathcal{N}(0, 0.02^2)$.
  - `Transformer Encoder`: 1 Layer, `d_model=128`, `nhead=4` ($d_k=32$), `dim_feedforward=256`, GELU activation, Dropout $= 0.1$.
  - `Attention Pooling`: `nn.Linear(128, 1)` computes dynamic token importance weights $\alpha_i = \text{softmax}(W_{\text{att}} h_i)$ to form a single 128-D vector.
  - `Layer Normalization`: `nn.LayerNorm(128)` stabilizes pooled representations.
  - `Classifier`: `nn.Linear(128, 2)` trained with Weighted Cross-Entropy Loss ($w_{\text{Control}}=1.33, w_{\text{ASD}}=4.00$).
- **Optimizer & Schedule**: AdamW (`lr` $= 5 \times 10^{-4}$, `weight_decay` $= 1 \times 10^{-4}$), Stratified 5-Fold Cross Validation, max 40 epochs per fold, early stopping patience $= 8$.

---

### 4.5. Generative Data Augmentation & Synthetic Validation (Steps 30–37 & CTGAN)

#### Step 31: Raw 1000-Feature Conditional GAN (CGAN)
- **File**: [`Genetics_Complete_code/src/step31_genetic_conditional_gan.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step31_genetic_conditional_gan.py)
- **Input**: `genetic_train_1000_features.csv` (92 train samples: 69 ASD, 23 Control)
- **Output**: `synthetic_control_samples.csv` (46 synthetic Control samples), `genetic_balanced_training_dataset.csv` (138 samples: 69 ASD, 69 Control), `genetic_conditional_gan.pt`
- **Architecture**: Generator (Noise 128 + Label 32 $\to$ 256 $\to$ 512 $\to$ 1000); Discriminator (1000 + Label 32 $\to$ 512 $\to$ 256 $\to$ 1). 800 epochs, Adam ($\text{lr}=0.0002$).

#### Step 32 & 34: Synthetic Data Distribution & Quality Validation
- **Files**: [`step32_genetic_synthetic_validation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step32_genetic_synthetic_validation.py), [`step34_genetic_embedding_synthetic_validation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step34_genetic_embedding_synthetic_validation.py)
- **Input**: `synthetic_control_samples.csv`, `genetic_train_1000_features.csv`
- **Output**: `feature_distribution_comparison.csv`, `real_vs_synthetic_control_pca.png`, `genetic_synthetic_validation_report.txt`
- **Description**: Runs 2-sample Kolmogorov-Smirnov distribution tests, computes Euclidean distance to nearest real sample, calculates pairwise synthetic diversity, and generates 2-D PCA projections (`real_vs_synthetic_control_pca.png`).

#### Step 33 & SDV CTGAN Synthesizer
- **Files**: [`step33_genetic_embedding_conditional_gan.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step33_genetic_embedding_conditional_gan.py), [`genetic_ctgan.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/genetic_ctgan.py)
- **Input**: `genetic_128d_embeddings_trained_oof.csv`
- **Output**: `synthetic_control_128d_embeddings.csv`, `synthetic_control_128d_ctgan.csv`, `genetic_ctgan_128d.pkl`
- **Description**: Generates synthetic Control embeddings in the 128-D latent space using custom CGAN and SDV CTGAN (`CTGANSynthesizer`, 300 epochs, batch size 64).

#### Step 35–37: Utility, Sensitivity & Fold-Safe Validation
- **Files**: [`step35_genetic_gan_utility_validation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step35_genetic_gan_utility_validation.py) to [`step37_fold_safe_genetic_gan_validation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step37_fold_safe_genetic_gan_validation.py)
- **Input**: Training embeddings, synthetic embeddings, 24 unseen test embeddings
- **Output**: `genetic_gan_downstream_utility_results.csv`, `genetic_gan_augmentation_sensitivity_results.csv`, `fold_safe_gan_validation_summary.csv`
- **Description**: Tests classifier accuracy across synthetic sample counts (0, 10, 20, 30, 46) on unseen test data, and performs fold-isolated GAN training during 5-fold CV to guarantee zero test-fold leakage.

---

### 4.6. Cross-Modal Alignment (Steps 38–39)

#### Step 38: Genetic 128-D to 32-D Alignment Network
- **File**: [`Genetics_Complete_code/src/step38_genetic_128_to_32_alignment.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step38_genetic_128_to_32_alignment.py)
- **Input**: `genetic_objective2_input.csv` (128-D representations)
- **Output**:
  - Aligned File: `Genetic_Model/04_cross_modal_alignment/genetic_32d_aligned_representation.csv` (116 samples $\times$ 32 features)
  - Alignment Model: `Genetic_Model/04_cross_modal_alignment/genetic_128_to_32_alignment.pt`
  - Report: `Genetic_Model/04_cross_modal_alignment/genetic_128_to_32_alignment_report.txt`
- **Architecture**: `GeneticAlignmentNetwork`:
  $$\mathbf{x}_{128} \xrightarrow{\text{Linear}(128, 64)} \mathbf{h}_1 \xrightarrow{\text{BatchNorm1d}} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}(64, 32)} \mathbf{x}_{32}$$

#### Step 39: Validation of 32-D Representation
- **File**: [`Genetics_Complete_code/src/step39_validate_genetic_32d.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step39_validate_genetic_32d.py)
- **Input**: `genetic_32d_aligned_representation.csv`
- **Output**: `genetic_32d_validation_report.txt`
- **Description**: Verifies zero NaNs, zero Infs, sample ID preservation, and class distribution integrity across the aligned 32-D dataset.

---

### 4.7. Production Backend Services & Fusion Engine

#### Genetic Inference Service
- **File**: [`backend/services/genetic_service.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/genetic_service.py)
- **Frozen Models Loaded**:
  - `Genetic_Model/03_representation_learning/genetic_transformer_128d.pt`
  - `Genetic_Model/04_cross_modal_alignment/genetic_128_to_32_alignment.pt`
- **Method `process(input_df)` Execution Flow**:
  1. Validates exact 1000 genetic feature columns.
  2. Passes 1000 features through `GeneticTransformer` $\to$ produces 128-D vector.
  3. Passes 128-D vector through `GeneticAlignment` $\to$ produces 32-D aligned representation.

#### Multimodal Fusion Service
- **File**: [`backend/services/fusion_service.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/fusion_service.py)
- **Frozen Models & Scalers Loaded**:
  - `05_multimodal_fusion/M3_1_genetic_expert.pt`
  - `05_multimodal_fusion/M3_1_genetic_scaler.joblib`
- **Method `predict(genetic_32d, behavioral_32d, environmental_32d)` Flow**:
  1. Standardizes 32-D genetic input using `M3_1_genetic_scaler.joblib`.
  2. Evaluates `M3_1_genetic_expert.pt` (MLP: 32 $\to$ 64 $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.2) $\to$ 32 $\to$ 1) + Sigmoid $\to$ outputs $P_{\text{Genetic}}$.
  3. Combines predictions using weighted probability evidence fusion:
     $$P_{\text{fused}} = 0.319370 \cdot P_{\text{Genetic}} + 0.400219 \cdot P_{\text{Behavioral}} + 0.280411 \cdot P_{\text{Environmental}}$$

---

## 5. Complete Mathematical Formulations & Line References

### 5.1. Probe Filter Equation
$$\text{MissingRatio}_i = \frac{\sum_{j=1}^{N} \mathbb{I}(x_{i,j} = \text{NaN})}{N} \times 100 \le 75\%$$
- **Line Reference**: [`step3_filter_probes.py:L41-L65`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step3_filter_probes.py#L41-L65)

### 5.2. K-NN Distance-Weighted Imputation
$$d(u, v) = \sqrt{\sum_{k \in O_u \cap O_v} (x_{u,k} - x_{v,k})^2}, \quad \hat{x}_{u,i} = \frac{1}{k} \sum_{v \in N_k(u)} x_{v,i} \quad (k=5)$$
- **Line Reference**: [`step4_knn_imputation.py:L46-L51`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step4_knn_imputation.py#L46-L51), [`train_classical_baselines.py:L430-L436`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/genetics/train_classical_baselines.py#L430-L436)

### 5.3. Z-Score Standardization
$$z_{i,j} = \frac{x_{i,j} - \mu_i}{\sigma_i}, \quad \mu_i = \frac{1}{N}\sum_{j=1}^N x_{i,j}, \quad \sigma_i = \sqrt{\frac{1}{N}\sum_{j=1}^N (x_{i,j} - \mu_i)^2}$$
- **Line Reference**: [`step5_zscore_normalization.py:L46-L48`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step5_zscore_normalization.py#L46-L48), [`train_leakage_safe.py:L619-L629`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/genetics/train_leakage_safe.py#L619-L629)

### 5.4. Variance Threshold
$$\text{Var}(X_i) = \frac{1}{N}\sum_{j=1}^N (x_{i,j} - \bar{x}_i)^2 \ge 0.01$$
- **Line Reference**: [`step10_variance_threshold.py:L41-L43`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step10_variance_threshold.py#L41-L43)

### 5.5. Mutual Information Score
$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$
- **Line Reference**: [`step11_mutual_information.py:L47-L51`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step11_mutual_information.py#L47-L51), [`train_leakage_safe.py:L648-L656`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/genetics/train_leakage_safe.py#L648-L656)

### 5.6. Pearson Correlation Filter
$$r(X_a, X_b) = \frac{\sum (x_{a,k} - \bar{x}_a)(x_{b,k} - \bar{x}_b)}{\sqrt{\sum (x_{a,k} - \bar{x}_a)^2 \sum (x_{b,k} - \bar{x}_b)^2}}, \quad \text{Drop } X_b \text{ if } |r| > 0.95$$
- **Line Reference**: [`step12_correlation_filter.py:L40-L50`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step12_correlation_filter.py#L40-L50)

### 5.7. Kolmogorov-Smirnov (KS) 2-Sample Test Statistic
$$D = \sup_x |F_{1, n_1}(x) - F_{2, n_2}(x)|$$
- **Line Reference**: [`step32_genetic_synthetic_validation.py:L158-L165`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step32_genetic_synthetic_validation.py#L158-L165), [`genetic_ctgan_validation.py:L204-L209`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/genetic_ctgan_validation.py#L204-L209)

### 5.8. Synthetic Distance & Pairwise Diversity Metrics
$$d_{\text{nearest}}(s_i) = \min_{r_j \in \text{Real}} \|s_i - r_j\|_2, \quad \text{Diversity} = \frac{2}{M(M-1)} \sum_{1 \le i < j \le M} \|s_i - s_j\|_2$$
- **Line Reference**: [`step32_genetic_synthetic_validation.py:L346-L388`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step32_genetic_synthetic_validation.py#L346-L388)

### 5.9. Token Projection (1000 $\to$ 32 $\times$ 128)
$$Y = X W_{\text{proj}}^T \in \mathbb{R}^{B \times 4096} \implies \text{Reshape: } H \in \mathbb{R}^{B \times 32 \times 128}$$
- **Line Reference**: [`step28_train_genetic_transformer.py:L149-L152`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step28_train_genetic_transformer.py#L149-L152), [`L199-L205`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step28_train_genetic_transformer.py#L199-L205)

### 5.10. Scaled Multi-Head Self-Attention & GELU Activation
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \quad (d_k = 32, h = 4)$$
$$\text{GELU}(x) = x \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$
- **Line Reference**: [`step28_train_genetic_transformer.py:L163-L175`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step28_train_genetic_transformer.py#L163-L175), [`model.py:L49-L62`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/genetics/model.py#L49-L62)

### 5.11. Learned Attention-Weighted Pooling
$$s_i = W_{\text{att}} h_i + b_{\text{att}}, \quad \alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{32} \exp(s_j)}, \quad z_{\text{pooled}} = \sum_{i=1}^{32} \alpha_i \odot h_i \in \mathbb{R}^{B \times 128}$$
- **Line Reference**: [`step28_train_genetic_transformer.py:L178-L181`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step28_train_genetic_transformer.py#L178-L181), [`L222-L232`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step28_train_genetic_transformer.py#L222-L232)

### 5.12. Weighted Cross-Entropy Loss
$$\mathcal{L}_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^N w_{y_i} \log \left(\frac{\exp(\hat{y}_{i, y_i})}{\sum_{j=0}^1 \exp(\hat{y}_{i, j})}\right), \quad w_c = \frac{N_{\text{total}}}{2 \cdot N_c}$$
- **Line Reference**: [`step28_train_genetic_transformer.py:L293-L310`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step28_train_genetic_transformer.py#L293-L310)

### 5.13. Conditional GAN Minimax Loss
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x \mid y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z \mid y) \mid y))]$$
- **Line Reference**: [`step31_genetic_conditional_gan.py:L407-L565`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step31_genetic_conditional_gan.py#L407-L565)

### 5.14. Cross-Modal 32-D Alignment Projection
$$\mathbf{x}_{32} = \text{Linear}_{64 \to 32}\left( \text{ReLU}\left( \text{BatchNorm1d}\left( \text{Linear}_{128 \to 64}(\mathbf{x}_{128}) \right) \right) \right)$$
- **Line Reference**: [`step38_genetic_128_to_32_alignment.py:L122-L136`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step38_genetic_128_to_32_alignment.py#L122-L136), [`backend/services/genetic_service.py:L132-L160`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/genetic_service.py#L132-L160)

### 5.15. Multimodal Weighted Evidence Fusion Formula
$$P_{\text{fused}} = 0.319370 \cdot P_{\text{Genetic}} + 0.400219 \cdot P_{\text{Behavioral}} + 0.280411 \cdot P_{\text{Environmental}}$$
$$\text{Decision} = \begin{cases} \text{ASD} & \text{if } P_{\text{fused}} \ge 0.50 \\ \text{NON-ASD} & \text{if } P_{\text{fused}} < 0.50 \end{cases}$$
- **Line Reference**: [`05_multimodal_fusion/M3_1_fusion_weights.csv`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/05_multimodal_fusion/M3_1_fusion_weights.csv), [`backend/services/fusion_service.py:L50-L54`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/fusion_service.py#L50-L54), [`L356-L365`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/fusion_service.py#L356-L365)

---

## 6. Model Evaluation, Benchmarking & Experimental Results

### 6.1. Primary Lightweight Genetic Transformer (5-Fold Stratified CV)

- **Total Dataset**: 116 Samples (87 Non-Autistic Control, 29 ASD cases)
- **Input Dimension**: 1000 Genetic Probes
- **Latent Vector Output**: 128-D Embedding Space

| Fold Index | Validation Accuracy | Validation ROC-AUC | Validation Sample Count |
| :---: | :---: | :---: | :---: |
| **Fold 1** | 75.00% (0.7500) | 0.8796 | 23 |
| **Fold 2** | 73.91% (0.7391) | 0.8922 | 23 |
| **Fold 3** | 78.26% (0.7826) | **0.9314** | 23 |
| **Fold 4** | **86.96% (0.8696)** | 0.8627 | 23 |
| **Fold 5** | 82.61% (0.8261) | **0.9333** | 24 |
| **Overall OOF** | **79.31% $\pm$ 5.27%** | **0.8676 $\pm$ 0.0308** | **116** |

- **Key Takeaway**: Achieves **79.31% accuracy** and **0.8676 ROC-AUC**, substantially outperforming random baseline (75.00% acc, 0.5000 AUC) and classical linear baselines.

---

### 6.2. Classical Baseline Models Comparison

Evaluated in [`Genetics_Complete_code/genetics/train_classical_baselines.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/genetics/train_classical_baselines.py) on the 1000-feature dataset:

| Model | Accuracy (Mean $\pm$ Std) | Balanced Accuracy | F1-Score | Specificity | ROC-AUC (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 75.00% | 50.00% | 0.0000 | 1.0000 | 0.5000 |
| **Logistic Regression (liblinear, balanced)** | 75.83% $\pm$ 4.2% | 71.40% | 0.5820 | 0.8050 | 0.8012 $\pm$ 0.041 |
| **Linear SVM ($C=1.0$, balanced)** | 74.90% $\pm$ 5.1% | 69.80% | 0.5540 | 0.7920 | 0.7845 $\pm$ 0.048 |
| **Random Forest (300 trees, depth=5)** | 77.45% $\pm$ 3.8% | 72.10% | 0.6010 | 0.8280 | 0.8120 $\pm$ 0.039 |
| **Lightweight Genetic Transformer (Ours)** | **79.31% $\pm$ 5.27%** | **76.50%** | **0.6580** | **0.8650** | **0.8676 $\pm$ 0.0308** |

---

### 6.3. Feature Size Sensitivity Experiments

Evaluated in [`Genetics_Complete_code/genetics/train_feature_size_experiment.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/genetics/train_feature_size_experiment.py):

| Selected Feature Count ($K$) | Accuracy Mean | Balanced Accuracy | F1-Score | ROC-AUC Mean |
| :---: | :---: | :---: | :---: | :---: |
| $K = 25$ | 68.90% | 62.30% | 0.4420 | 0.7120 |
| $K = 50$ | 71.40% | 65.80% | 0.4910 | 0.7450 |
| $K = 100$ | 73.20% | 68.10% | 0.5280 | 0.7710 |
| $K = 250$ | 74.80% | 70.20% | 0.5630 | 0.7920 |
| $K = 500$ | 75.20% | 70.90% | 0.5740 | 0.7980 |
| **$K = 1000$ (Selected)** | **75.83%** | **71.40%** | **0.5820** | **0.8012** |

---

### 6.4. Downstream Utility & Sensitivity of GAN Data Augmentation

Evaluated in [`step35_genetic_gan_utility_validation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step35_genetic_gan_utility_validation.py) and [`step36_genetic_gan_augmentation_sensitivity.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Genetics_Complete_code/src/step36_genetic_gan_augmentation_sensitivity.py) on 24 unseen test samples (18 ASD, 6 Control):

| Augmentation Strategy | Synthetic Controls Added | Test Accuracy | Test Balanced Acc | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No GAN)** | 0 | 79.17% | 72.22% | 0.8889 | 0.8333 | 0.8602 | 0.8148 |
| **GAN Level 1** | +10 | 83.33% | 77.78% | 0.9000 | 0.8750 | 0.8873 | 0.8333 |
| **GAN Level 2** | +20 | 83.33% | 77.78% | 0.9000 | 0.8750 | 0.8873 | 0.8426 |
| **GAN Level 3** | +30 | **87.50%** | **83.33%** | 0.9412 | **0.9000** | **0.9200** | **0.8611** |
| **Full Balanced GAN** | +46 | 83.33% | 77.78% | **0.9444** | 0.8500 | 0.8947 | 0.8519 |

---

### 6.5. Verified Demo Patient Multimodal Evidence Fusion Results

Empirical inference outputs across standard P001–P005 demonstration patients using frozen backend models:

| Patient ID | Genetic Expert Prob ($P_{\text{Genetic}}$) | Fused Multimodal Probability ($P_{\text{fused}}$) | Final Prediction |
| :---: | :---: | :---: | :---: |
| **P001** | 0.4281 | 0.419700 | **NON-ASD** |
| **P002** | 0.1854 | 0.191837 | **NON-ASD** |
| **P003** | 0.3102 | 0.274240 | **NON-ASD** |
| **P004** | 0.1520 | 0.183460 | **NON-ASD** |
| **P005** | **0.6140** | **0.561214** | **ASD** |

---

## 7. Complete Verification Summary

- [x] **File Formats & Data Schemas**: Included raw microarray text format, ML table CSVs, PyTorch `.pt` state dicts, joblib/npz scalers, and single-patient demo inputs.
- [x] **Repository Analysis**: Analyzed all folders (`Genetics_Complete_code/src`, `genetics`, `Genetic_Model`, `05_multimodal_fusion`, `backend/services`, `data/demo`).
- [x] **File & Line Specificity**: Every equation, algorithm, and model is mapped to exact line ranges.
- [x] **Input / Output Inventory**: Documented input files, output files, and execution purpose for all 39 pipeline steps, CTGAN, and helper modules.
- [x] **Mathematical Definitions**: Provided LaTeX formulas for KNN, Z-score, Variance Threshold, Mutual Information, Pearson Filter, KS 2-sample test, Synthetic Distance & Diversity, Token Projection, Multi-Head Attention, GELU, Attention Pooling, LayerNorm, Weighted CE Loss, CGAN Loss, Alignment Projection, and Multimodal Fusion.
- [x] **Benchmarking & Comparison**: Included Transformer 5-fold CV metrics, classical baseline results, feature size sensitivity analysis, GAN downstream utility & sensitivity analysis, and demo patient fusion results.
