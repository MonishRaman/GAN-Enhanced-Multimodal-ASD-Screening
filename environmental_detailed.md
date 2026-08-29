# Complete & Exhaustive Environmental Modality Analysis: Architecture, Mathematical Formulations, Pipeline Flow, Input/Output Inventory, File Formats, and Benchmarks

---

## 1. Repository & Directory Inventory

The environmental modality in the **GAN-Enhanced Multimodal ASD Screening** project processes population-scale pediatric health metrics from the **National Survey of Children's Health (NSCH 2023 Topical Dataset)**, performing feature cleaning, categorical encoding, statistical feature selection, deep representation learning (128-D embedding space), cross-modal 32-D alignment projection, and production backend service integration.

```
d:\GAN-Enhanced-Multimodal-ASD-Screening\
├── Environmental_complete_code_directory/ # End-to-end execution scripts & dataset files
│   ├── src/                               # 17-step pipeline execution scripts
│   │   ├── step1.py                       # Initial NSCH dataset inspection
│   │   ├── step2_clean.py                 # Target definition (K2Q35A) & basic missing cleanup
│   │   ├── step3_inspect.py               # Column data type & distribution inspection
│   │   ├── step4_preprocess.py            # Train/test split, ColumnTransformer, encoding
│   │   ├── step5_feature_selection.py     # Mutual Information feature scoring
│   │   ├── step5b_group_features.py       # Domain-based feature grouping & aggregation
│   │   ├── step6_model_feature_selection.py # Random Forest importance feature selection
│   │   ├── step6b_group_rf.py             # Random Forest grouped importance analysis
│   │   ├── step7_select_features.py       # Feature selection consensus (Top 20/31 features)
│   │   ├── step8_prepare_selected.py      # Final selected tabular dataset creation
│   │   ├── step9_environmental_representation.py # Keras 128-D representation encoder
│   │   ├── step10_threshold_analysis.py   # Decision threshold & F1 optimization
│   │   ├── step11_validate_embeddings.py  # 128-D embedding representation validation
│   │   ├── step12_reduce_embedding.py     # PCA / Autoencoder 128-D to 32-D dimension reduction
│   │   ├── step13_validate_32d.py         # 32-D aligned vector verification
│   │   ├── step14_add_target_to_32d.py    # Target label re-attachment for downstream evaluation
│   │   └── step15_evaluate_32d.py         # Final 32-D classifier evaluation & metrics
│   ├── nsch_2023e_topical.csv             # Raw NSCH 2023 survey dataset (69,043,580 bytes)
│   ├── environmental_dataset.csv          # Preprocessed environmental feature table
│   ├── environmental_selected_20.csv     # Selected 31 encoded environmental variables
│   ├── environmental_embedding_train.csv # 128-D training embedding table
│   ├── environmental_embedding_test.csv  # 128-D test embedding table
│   ├── environmental_embedding_train_32d.csv # 32-D training representation vector
│   ├── environmental_embedding_test_32d.csv  # 32-D test representation vector
│   ├── environmental_representation_model.keras # Frozen TensorFlow/Keras representation model
│   └── environmental_pca_128_to_32.joblib # Dimensionality reduction joblib artifact
├── Environmental_Model/                  # Frozen model checkpoints for production backend
│   ├── 01_raw_data/                       # Cleaned input data table references
│   ├── 02_preprocessing/
│   │   └── environmental_preprocessor.joblib # Frozen Scikit-Learn ColumnTransformer pipeline
│   ├── 03_representation_learning/
│   │   ├── environmental_encoder_128d.pt  # Frozen PyTorch 128-D Encoder weights
│   │   ├── environmental_128d_embeddings.csv # Full population 128-D embedding matrix
│   │   └── environmental_128d_training_report.txt # Official training report
│   └── 04_cross_modal_alignment/
│       ├── environmental_128_to_32_alignment.pt # Frozen 128-D to 32-D Autoencoder Alignment weights
│       ├── environmental_32d_aligned_representation.csv # Final 32-D aligned vector table
│       └── environmental_128_to_32_alignment_report.txt # Alignment validation report
├── 05_multimodal_fusion/                 # Knowledge-level fusion directory
│   ├── M3_1_environmental_expert.pt      # Environmental Expert NN classifier checkpoint
│   ├── M3_1_environmental_scaler.joblib  # StandardScaler for 32-D environmental input
│   └── M3_1_fusion_weights.csv           # Multimodal fusion weights (Environmental = 0.280411)
├── backend/
│   └── services/
│       └── environmental_service.py      # Production FastAPI service for local environmental inference
└── MODEL_MANIFEST.json                   # System manifest freezing ML artifacts
```

---

## 2. File Formats & Technical Schemas

| Category | File Format | Path / Location | Description & Schema |
| :--- | :--- | :--- | :--- |
| **Raw Survey Dataset** | Tabular `.csv` | `Environmental_complete_code_directory/nsch_2023e_topical.csv` | 54,106 Rows $\times$ 568 Columns (Demographics, household income, ACEs, healthcare access). |
| **Target Clean Dataset** | Matrix `.csv` | `.../environmental_step2_target_clean.csv` | 47,768 Rows (valid `K2Q35A` responses) $\times$ 33 Features (`K2Q35A` binary target + 32 environment variables). |
| **Preprocessor Artifact** | `.joblib` binary | `Environmental_Model/02_preprocessing/environmental_preprocessor.joblib` | Scikit-Learn `ColumnTransformer` (OneHotEncoder for categorical, SimpleImputer + StandardScaler for numeric). |
| **Selected Feature Table** | Header `.csv` | `.../environmental_selected_20.csv` | 47,768 Rows $\times$ 31 Features (One-hot encoded numeric/categorical environmental predictors). |
| **PyTorch Encoder Checkpoint** | Checkpoint `.pt` | `Environmental_Model/03_representation_learning/environmental_encoder_128d.pt` | Binary PyTorch `state_dict` mapping `encoder.0`, `encoder.1`, `encoder.4`, `encoder.5`, `classifier`. |
| **PyTorch Alignment Checkpoint** | Checkpoint `.pt` | `Environmental_Model/04_cross_modal_alignment/environmental_128_to_32_alignment.pt` | Binary PyTorch `state_dict` mapping `encoder` (128 $\to$ 64 $\to$ 32) and `decoder` (32 $\to$ 64 $\to$ 128). |
| **128-D Embedding Table** | Header `.csv` | `Environmental_Model/03_representation_learning/environmental_128d_embeddings.csv` | 47,768 Rows $\times$ 128 Columns (`env_emb_000` .. `env_emb_127`). |
| **32-D Aligned Table** | Header `.csv` | `Environmental_Model/04_cross_modal_alignment/environmental_32d_aligned_representation.csv` | 47,768 Rows $\times$ 32 Columns (`env_aligned_000` .. `env_aligned_031`). |

---

## 3. End-to-End Pipeline Flow Diagram & Overview

```mermaid
flowchart TD
    A["Raw NSCH 2023 Survey Dataset (54,106 Samples × 568 Columns)"] --> B["Step 1-3: Target Filtering & Missing Value Cleaning (Target: K2Q35A)"]
    B --> C["Step 4: Preprocessing (Categorical One-Hot & Numeric Standardization)"]
    C --> D["Step 5-7: Feature Selection (Mutual Information & Random Forest Importance -> 31 Features)"]
    D --> E["Step 8-9: Deep Representation Learning (31 -> 128-D Latent Embedding Space)"]
    E --> F1["Step 10: Threshold Optimization (Max F1-Score on Validation Split)"]
    E --> F2["Step 12: Cross-Modal Alignment Autoencoder (128-D -> 64 -> 32-D Aligned Vector)"]
    F2 --> G["Step 13-15: Representation Quality & Classifier Evaluation"]
    F2 --> H["Backend Service: Production Single-Patient Environmental Inference Engine"]
    H --> I["Multimodal Evidence Fusion (Weight = 0.280411)"]
```

---

## 4. Exhaustive Step-by-Step Code, Model, and Input/Output Inventory

### 4.1. Data Ingestion, Cleaning & Preprocessing (Steps 1–4)

#### Step 1 & 2: Dataset Inspection & Target Definition
- **Files**: [`Environmental_complete_code_directory/src/step1.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step1.py), [`Environmental_complete_code_directory/src/step2_clean.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step2_clean.py)
- **Input**: `nsch_2023e_topical.csv`
- **Output**: `environmental_step2_target_clean.csv`
- **Description**: Filters survey responses for item `K2Q35A` ("Has a doctor or healthcare provider ever told you that this child has Autism or Autism Spectrum Disorder?"). Converts response codes $1 \to 1$ (ASD positive) and $2 \to 0$ (Control). Drops invalid/missing target rows ($N = 47,768$ retained samples).

#### Step 3: Column Type Inspection & Sparse Filtering
- **File**: [`Environmental_complete_code_directory/src/step3_inspect.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step3_inspect.py)
- **Input**: `environmental_step2_target_clean.csv`
- **Output**: Console diagnostics reporting column types and missing value counts.
- **Description**: Inspects candidate predictor variables across demographic, socioeconomic, and environmental domains.

#### Step 4: Preprocessing Pipeline Construction
- **File**: [`Environmental_complete_code_directory/src/step4_preprocess.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step4_preprocess.py)
- **Input**: `environmental_step2_target_clean.csv`
- **Output**: `X_train_environmental.csv`, `X_test_environmental.csv`, `y_train_environmental.csv`, `y_test_environmental.csv`, `Environmental_Model/02_preprocessing/environmental_preprocessor.joblib`
- **Description**: Drops extremely sparse feature `k9q41`. Identifies 29 categorical variables (e.g., `higrade`, `family_r`, `house_gen`, `metro_yn`, `foodsit`, `ace1`..`ace10`, `screentime`) and 2 numeric variables (`fpl_i1` federal poverty line ratio, `hoursleep`). Applies stratified 80/20 train-test split ($38,214$ train / $9,554$ test). Builds Scikit-Learn `ColumnTransformer` executing `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features and `Pipeline([SimpleImputer(strategy='median'), StandardScaler()])` for numeric features.

---

### 4.2. Feature Selection & Consensus Scoring (Steps 5–8)

#### Step 5 & 5b: Mutual Information Feature Selection
- **Files**: [`Environmental_complete_code_directory/src/step5_feature_selection.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step5_feature_selection.py), [`step5b_group_features.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step5b_group_features.py)
- **Input**: `X_train_environmental.csv`, `y_train_environmental.csv`
- **Output**: `environmental_feature_importance.csv`, `environmental_grouped_feature_importance.csv`
- **Description**: Computes Mutual Information scores using `mutual_info_classif(random_state=42)` across encoded features. Aggregates one-hot dummy column scores back to primary survey variable groups.

#### Step 6 & 6b: Random Forest Feature Selection
- **Files**: [`Environmental_complete_code_directory/src/step6_model_feature_selection.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step6_model_feature_selection.py), [`step6b_group_rf.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step6b_group_rf.py)
- **Input**: `X_train_environmental.csv`, `y_train_environmental.csv`
- **Output**: `environmental_rf_feature_importance.csv`, `environmental_grouped_rf_importance.csv`
- **Description**: Fits a Random Forest Classifier ($500$ trees, `class_weight='balanced'`) to compute Gini feature importance rankings.

#### Step 7 & 8: Consensus Selection & Final Table Assembly
- **Files**: [`Environmental_complete_code_directory/src/step7_select_features.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step7_select_features.py), [`step8_prepare_selected.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step8_prepare_selected.py)
- **Input**: Importance ranking `.csv` files, preprocessed feature matrices
- **Output**: `environmental_selected_20.csv`, `X_train_selected20.csv`, `X_test_selected20.csv`
- **Description**: Selects the top 31 encoded environmental predictor variables exhibiting high mutual information and tree Gini importance (covering household poverty level, parental education, Adverse Childhood Experiences, neighborhood physical environment, and daily screen time).

---

### 4.3. Representation Learning — 128-D Environmental Encoder (Step 9)

#### Step 9: Environmental Representation Encoder Training
- **File**: [`Environmental_complete_code_directory/src/step9_environmental_representation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py)
- **Input**: `X_train_selected20.csv` ($38,214 \times 31$), `X_test_selected20.csv` ($9,554 \times 31$)
- **Output**:
  - Model Checkpoint: `environmental_representation_model.keras`
  - PyTorch Checkpoint: `Environmental_Model/03_representation_learning/environmental_encoder_128d.pt`
  - Embedding Matrices: `environmental_embedding_train.csv`, `environmental_embedding_test.csv` ($47,768 \times 128$)
  - Technical Report: `Environmental_Model/03_representation_learning/environmental_128d_training_report.txt`
- **Architecture**: `EnvironmentalEncoder`:
  - `Input Layer`: Input shape $(31,)$
  - `Dense Layer 1`: `Linear(31, 128)` $\to$ `BatchNorm1d(128)` $\to$ `ReLU()` $\to$ `Dropout(0.20)`
  - `Dense Layer 2`: `Linear(128, 128)` $\to$ `BatchNorm1d(128)` $\to$ `ReLU()` $\to$ `Dropout(0.20)`
  - `Embedding Bottleneck`: 128-dimensional linear representation vector $\mathbf{z}_{\text{env}} \in \mathbb{R}^{128}$
  - `Classifier Head`: `Linear(128, 1)` $\to$ `Sigmoid`
- **Training Parameters**: Loss = Binary Cross-Entropy, Optimizer = Adam ($\text{lr}=1 \times 10^{-3}$), Class Weights ($w_0 = 0.52, w_1 = 11.85$), Epochs = 50, Early Stopping Patience = 10.

---

### 4.4. Cross-Modal Alignment — 128-D to 32-D Projection (Steps 12–15)

#### Step 12 & 13: Autoencoder Alignment Projection & Validation
- **Files**: [`Environmental_complete_code_directory/src/step12_reduce_embedding.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step12_reduce_embedding.py), [`step13_validate_32d.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step13_validate_32d.py)
- **Input**: `environmental_128d_embeddings.csv` ($47,768 \times 128$)
- **Output**:
  - Aligned File: `Environmental_Model/04_cross_modal_alignment/environmental_32d_aligned_representation.csv` ($47,768 \times 32$)
  - Model Checkpoint: `Environmental_Model/04_cross_modal_alignment/environmental_128_to_32_alignment.pt`
  - Report: `Environmental_Model/04_cross_modal_alignment/environmental_128_to_32_alignment_report.txt`
- **Architecture**: `EnvironmentalAlignment`:
  - `Encoder Subnetwork`:
    $$\mathbf{x}_{128} \xrightarrow{\text{Linear}(128, 64)} \mathbf{h}_1 \xrightarrow{\text{BatchNorm1d}} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}(64, 32)} \mathbf{x}_{32}$$
  - `Decoder Subnetwork`:
    $$\mathbf{x}_{32} \xrightarrow{\text{Linear}(32, 64)} \mathbf{h}_2 \xrightarrow{\text{BatchNorm1d}} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}(64, 128)} \hat{\mathbf{x}}_{128}$$
- **Reconstruction MSE**: $0.135754$. Target labels are strictly unobserved during alignment training.

#### Step 14 & 15: Aligned Representation Evaluation
- **Files**: [`Environmental_complete_code_directory/src/step14_add_target_to_32d.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step14_add_target_to_32d.py), [`step15_evaluate_32d.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step15_evaluate_32d.py)
- **Input**: `environmental_embedding_test_32d_with_target.csv`
- **Output**: `environmental_32d_performance.csv`, `environmental_32d_roc_curve.png`, `environmental_32d_precision_recall_curve.png`
- **Description**: Verifies that 32-D compressed environmental representations preserve complete diagnostic predictive power relative to the 128-D embedding space.

---

### 4.5. Production Backend Service & Multimodal Integration

#### Environmental Inference Service
- **File**: [`backend/services/environmental_service.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py)
- **Frozen Models Loaded**:
  - `Environmental_Model/02_preprocessing/environmental_preprocessor.joblib`
  - `Environmental_Model/03_representation_learning/environmental_encoder_128d.pt`
  - `Environmental_Model/04_cross_modal_alignment/environmental_128_to_32_alignment.pt`
- **Execution Workflow**:
  1. Validates exact environmental input feature schema.
  2. Applies `environmental_preprocessor.joblib` to transform raw survey responses $\to$ 31-D feature vector.
  3. Passes 31 features into `EnvironmentalEncoder.encode()` $\to$ outputs 128-D vector.
  4. Passes 128-D vector into `EnvironmentalAlignment()` $\to$ outputs 32-D aligned vector.

---

## 5. Complete Mathematical Formulations & Line References

### 5.1. Class-Weighted Binary Cross-Entropy Loss
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \log \hat{y}_i + w_0 (1 - y_i) \log (1 - \hat{y}_i) \right]$$
- **Line Reference**: [`Environmental_complete_code_directory/src/step9_environmental_representation.py:L70-L84`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py#L70-L84), [`backend/services/environmental_service.py:L98-L128`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py#L98-L128)

### 5.2. Mutual Information Feature Selection
$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$
- **Line Reference**: [`step5_feature_selection.py:L29-L34`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step5_feature_selection.py#L29-L34)

### 5.3. Random Forest Gini Impurity Reduction
$$G = 1 - \sum_{k=1}^K p_k^2, \quad \Delta G(m) = G_{\text{parent}} - \frac{N_{\text{left}}}{N} G_{\text{left}} - \frac{N_{\text{right}}}{N} G_{\text{right}}$$
- **Line Reference**: [`step6_model_feature_selection.py:L35-L48`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step6_model_feature_selection.py#L35-L48)

### 5.4. Batch Normalization Transformation
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2, \quad \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
- **Line Reference**: [`backend/services/environmental_service.py:L100-L107`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py#L100-L107)

### 5.5. Autoencoder Reconstruction Mean Squared Error (MSE) Loss
$$\mathcal{L}_{\text{recon}} = \frac{1}{N \cdot D} \sum_{i=1}^N \sum_{j=1}^D (x_{i,j} - \hat{x}_{i,j})^2, \quad (D = 128)$$
- **Line Reference**: [`Environmental_Model/04_cross_modal_alignment/environmental_128_to_32_alignment_report.txt:L37-L38`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_Model/04_cross_modal_alignment/environmental_128_to_32_alignment_report.txt#L37-L38), [`backend/services/environmental_service.py:L163-L190`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py#L163-L190)

### 5.6. Environmental Expert Classifier Architecture & Scaler
$$\mathbf{z}_{\text{scaled}} = \frac{\mathbf{x}_{32} - \boldsymbol{\mu}_{\text{env}}}{\boldsymbol{\sigma}_{\text{env}}}$$
$$P_{\text{Environmental}} = \sigma \left( W_3 \cdot \text{ReLU}\left( W_2 \cdot \text{Dropout}_{0.2}\left( \text{ReLU}\left( \text{BatchNorm1d}\left( W_1 \mathbf{z}_{\text{scaled}} + b_1 \right) \right) \right) + b_2 \right) + b_3 \right)$$
- **Line Reference**: [`05_multimodal_fusion/M3_1_environmental_expert.pt`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/05_multimodal_fusion/M3_1_environmental_expert.pt), [`backend/services/fusion_service.py:L63-L101`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/fusion_service.py#L63-L101)

---

## 6. Model Evaluation, Benchmarking & Experimental Results

### 6.1. Environmental 128-D Representation Learning Performance

- **Dataset Size**: 47,768 Survey Samples (32,481 Train, 5,733 Validation, 9,554 Unseen Test)
- **Input Dimension**: 31 Selected Encoded Features
- **Target Item**: `K2Q35A` (Autism Spectrum Disorder Healthcare Provider Diagnosis)

| Metric | Validation Split | Unseen Test Split |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.6924** | **0.6737** |
| **Accuracy** | 0.6312 | 0.6268 |
| **Balanced Accuracy** | 0.6315 | 0.6270 |
| **Precision** | 0.0742 | 0.0724 |
| **Recall** | 0.6310 | 0.6274 |
| **F1-Score** | 0.1328 | 0.1298 |

---

### 6.2. 128-D vs 32-D Aligned Vector Performance Comparison

Evaluated in `Environmental_complete_code_directory/src/step15_evaluate_32d.py`:

| Representation Dimension | Test ROC-AUC | Reconstruction MSE | Vector Dimension | NaN/Inf Count |
| :--- | :---: | :---: | :---: | :---: |
| **Uncompressed 128-D Embedding** | 0.6737 | N/A | 128 | 0 |
| **Aligned 32-D Vector (Ours)** | **0.6729** | **0.13575** | **32** | **0** |

- **Key Finding**: Compressing the representation from 128-D to 32-D preserves **99.88%** of the original 128-D diagnostic ROC-AUC while reducing storage and memory footprint by $75\%$.

---

## 7. Complete Verification Summary

- [x] **File Formats & Data Schemas**: Documented NSCH raw survey data, preprocessed tables, ColumnTransformers, PyTorch `.pt` checkpoints, and aligned CSV matrices.
- [x] **Repository Analysis**: Analyzed `Environmental_complete_code_directory/src`, `Environmental_Model`, `05_multimodal_fusion`, and `backend/services/environmental_service.py`.
- [x] **File & Line Specificity**: Formulations and architectures mapped directly to code lines.
- [x] **Input / Output Inventory**: Cataloged all 17 pipeline execution steps, feature selection scripts, representation trainers, autoencoder aligners, and backend inference handlers.
- [x] **Mathematical Formulations**: Provided LaTeX equations for Class-Weighted BCE, Mutual Information, Gini Impurity, BatchNorm1d, Autoencoder Reconstruction MSE, and Expert MLP evaluation.
- [x] **Benchmarking & Validation**: Included validation/test ROC-AUC metrics, 128-D vs 32-D preservation comparisons, and reconstruction quality scores.
