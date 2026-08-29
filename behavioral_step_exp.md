# Exhaustive Step-by-Step Mathematical & Formula Analysis: Behavioral Modality

---

## 1. Data Cleaning & Leakage Prevention (Step B1)

### 1.1. `Behavioral_Model/02_preprocessing/`
- **Files**: [`Behavioral_Model/01_raw_data/behavioral_dataset.csv`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/01_raw_data/behavioral_dataset.csv), [`Behavioral_Model/02_preprocessing/behavioral_cleaned_no_leakage.csv`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/02_preprocessing/behavioral_cleaned_no_leakage.csv)
- **Input**: Raw behavioral observation table ($N=880$ samples).
- **Formulas & Mathematical Logic**:
  - **Diagnostic Leakage Filter**:
    $$X_{\text{cleaned}} = X_{\text{raw}} \setminus \{ \text{Diagnostic Proxy Columns} \}$$
    - *Line Reference*: [`Behavioral_Model/02_preprocessing/behavioral_preprocessing_report.txt:L1-L15`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/02_preprocessing/behavioral_preprocessing_report.txt#L1-L15)
  - **Target Binary Encoding**:
    $$y_i = \begin{cases} 1 & \text{if ASD Positive} \\ 0 & \text{if Non-ASD Control} \end{cases}, \quad N_{\text{total}} = 880 \text{ samples}$$
    - *Line Reference*: [`Behavioral_Model/02_preprocessing/behavioral_preprocessing_report.txt:L10-L20`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/02_preprocessing/behavioral_preprocessing_report.txt#L10-L20)

---

## 2. Preprocessing & Sklearn Compatibility Shim (Step B2)

### 2.1. `Behavioral_Model/02_preprocessing/behavioral_preprocessor.joblib`
- **File**: [`Behavioral_Model/02_preprocessing/behavioral_preprocessor.joblib`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/02_preprocessing/behavioral_preprocessor.joblib)
- **Output**: `behavioral_train_processed.csv` ($704 \times 87$), `behavioral_test_processed_unseen.csv` ($176 \times 87$)
- **Formulas & Mathematical Logic**:
  - **Numeric Feature Standardization**:
    $$z_j = \frac{x_j - \mu_j}{\sigma_j}, \quad \mu_j = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} x_{i, j}, \quad \sigma_j = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (x_{i, j} - \mu_j)^2}$$
    - *Line Reference*: [`Behavioral_Model/02_preprocessing/behavioral_processed_feature_names.txt:L1-L87`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/02_preprocessing/behavioral_processed_feature_names.txt#L1-L87)
  - **Categorical Feature One-Hot Encoding**:
    $$\mathbf{x}_{\text{encoded}} = [ \mathbb{I}(x = c_1), \mathbb{I}(x = c_2), \dots, \mathbb{I}(x = c_K) ]^T \implies 87 \text{ total features}$$
    - *Line Reference*: [`Behavioral_Model/02_preprocessing/behavioral_processed_feature_names.txt:L1-L87`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/02_preprocessing/behavioral_processed_feature_names.txt#L1-L87)

---

### 2.2. Runtime Compatibility Shim (`backend/services/behavioral_service.py`)
- **File**: [`backend/services/behavioral_service.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py)
- **Formulas & Mathematical Logic**:
  - **Dynamic Private Class Invalidation Fallback**:
    $$\text{If } \neg \text{hasattr}(\text{column\_transformer}, \text{"\_RemainderColsList"}) \implies \text{Inject } \text{\_RemainderColsList} \in \text{type}(\text{list})$$
    - *Line Reference*: [`backend/services/behavioral_service.py:L72-L117`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L72-L117)

---

## 3. Deep Representation Learning (Step B3)

### 3.1. `Behavioral_Model/03_representation_learning/behavioral_encoder_128d.pt`
- **Files**: [`Behavioral_Model/03_representation_learning/behavioral_encoder_128d.pt`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/03_representation_learning/behavioral_encoder_128d.pt), [`backend/services/behavioral_service.py:L123-L178`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L123-L178)
- **Output**: `behavioral_128d_embeddings.csv` ($880 \times 128$), `behavioral_test_predictions.csv`
- **Formulas & Mathematical Logic**:
  - **Layer 1 Transformation**:
    $$\mathbf{h}_1 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_1 \mathbf{x}_{87} + b_1 \right) \right) \in \mathbb{R}^{128}, \quad \mathbf{h}_{1, \text{drop}} = \text{Dropout}_{0.2}(\mathbf{h}_1)$$
    - *Line Reference*: [`backend/services/behavioral_service.py:L163-L167`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L163-L167)
  - **Layer 2 Transformation**:
    $$\mathbf{h}_2 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_2 \mathbf{h}_{1, \text{drop}} + b_2 \right) \right) \in \mathbb{R}^{128}, \quad \mathbf{z}_{\text{beh}} = \text{Dropout}_{0.2}(\mathbf{h}_2) \in \mathbb{R}^{128}$$
    - *Line Reference*: [`backend/services/behavioral_service.py:L169-L173`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L169-L173)
  - **Auxiliary Output Classifier**:
    $$\hat{y} = \sigma\left( W_{\text{cls}} \mathbf{z}_{\text{beh}} + b_{\text{cls}} \right) = \frac{1}{1 + e^{-(W_{\text{cls}} \mathbf{z}_{\text{beh}} + b_{\text{cls}})}}$$
    - *Line Reference*: [`backend/services/behavioral_service.py:L150-L158`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L150-L158)
  - **Binary Cross-Entropy Training Objective**:
    $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]$$
    - *Line Reference*: [`Behavioral_Model/03_representation_learning/behavioral_128d_training_report.txt:L1-L21`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/03_representation_learning/behavioral_128d_training_report.txt#L1-L21)

---

## 4. Cross-Modal 32-D Autoencoder Alignment (Steps B4–B5)

### 4.1. `Behavioral_Model/04_cross_modal_alignment/behavioral_128_to_32_alignment.pt`
- **Files**: [`Behavioral_Model/04_cross_modal_alignment/behavioral_128_to_32_alignment.pt`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/04_cross_modal_alignment/behavioral_128_to_32_alignment.pt), [`backend/services/behavioral_service.py:L181-L245`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L181-L245)
- **Output**: `behavioral_32d_aligned_representation.csv` ($880 \times 32$)
- **Formulas & Mathematical Logic**:
  - **Encoder Subnetwork Mapping**:
    $$\mathbf{h}_1 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_1 \mathbf{x}_{128} + b_1 \right) \right) \in \mathbb{R}^{64}$$
    $$\mathbf{x}_{32} = W_2 \mathbf{h}_1 + b_2 \in \mathbb{R}^{32}$$
    - *Line Reference*: [`backend/services/behavioral_service.py:L217-L222`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L217-L222)
  - **Decoder Subnetwork Mapping**:
    $$\mathbf{h}_2 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_3 \mathbf{x}_{32} + b_3 \right) \right) \in \mathbb{R}^{64}$$
    $$\hat{\mathbf{x}}_{128} = W_4 \mathbf{h}_2 + b_4 \in \mathbb{R}^{128}$$
    - *Line Reference*: [`backend/services/behavioral_service.py:L228-L233`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/behavioral_service.py#L228-L233)
  - **Reconstruction MSE Loss**:
    $$\mathcal{L}_{\text{recon}} = \frac{1}{128 N} \sum_{i=1}^N \|\mathbf{x}_{128, i} - \hat{\mathbf{x}}_{128, i}\|_2^2 = 0.00071775$$
    - *Line Reference*: [`Behavioral_Model/04_cross_modal_alignment/behavioral_128_to_32_alignment_report.txt:L9`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Behavioral_Model/04_cross_modal_alignment/behavioral_128_to_32_alignment_report.txt#L9)

---

## 5. Behavioral Forensics & Column Audits (Step B6)

### 5.1. `05_multimodal_fusion/M4_1_2B_behavioral_forensics/`
- **Files**: [`05_multimodal_fusion/M4_1_2B_behavioral_forensics/M4_1_2B_behavioral_forensic_report.txt`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/05_multimodal_fusion/M4_1_2B_behavioral_forensics/M4_1_2B_behavioral_forensic_report.txt)
- **Formulas & Mathematical Logic**:
  - **Column Mean & Standard Deviation Integrity**:
    $$\bar{\mathbf{z}}_k = \frac{1}{880} \sum_{i=1}^{880} \mathbf{z}_{i, k}, \quad s_k = \sqrt{\frac{1}{879} \sum_{i=1}^{880} (\mathbf{z}_{i, k} - \bar{\mathbf{z}}_k)^2}, \quad k \in \{1, \dots, 128\}$$
    - *Line Reference*: [`05_multimodal_fusion/M4_1_2B_behavioral_forensics/behavioral_128d_numeric_column_statistics.csv:L1-L129`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/05_multimodal_fusion/M4_1_2B_behavioral_forensics/behavioral_128d_numeric_column_statistics.csv#L1-L129)
  - **Distributional Shift Test**:
    $$D_{\text{KS}} = \sup_x |F_{\text{train}}(x) - F_{\text{unseen}}(x)| < D_{\text{critical}}$$
    - *Line Reference*: [`05_multimodal_fusion/M4_1_2B_behavioral_forensics/M4_1_2B_behavioral_forensic_report.txt:L15-L35`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/05_multimodal_fusion/M4_1_2B_behavioral_forensics/M4_1_2B_behavioral_forensic_report.txt#L15-L35)
