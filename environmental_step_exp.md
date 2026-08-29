# Exhaustive Step-by-Step Mathematical & Formula Analysis: Environmental Modality

---

## 1. Data Ingestion, Target Cleaning & Inspection (Steps 1–3)

### 1.1. `step1.py` & `step2_clean.py`
- **Files**: [`Environmental_complete_code_directory/src/step1.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step1.py), [`step2_clean.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step2_clean.py)
- **Input**: Raw NSCH 2023 survey dataset `nsch_2023e_topical.csv` (54,106 samples $\times$ 568 items).
- **Description**: Inspects column schemas, maps primary ASD diagnostic item `K2Q35A`, filters missing survey responses, and encodes target labels into binary indicators.
- **Formulas & Mathematical Logic**:
  - **Target Binary Mapping**:
    $$y_i = \begin{cases} 1 & \text{if } \text{K2Q35A}_i = 1 \text{ (Autism/ASD Diagnosis)} \\ 0 & \text{if } \text{K2Q35A}_i = 2 \text{ (No Autism Diagnosis)} \end{cases}$$
    - *Line Reference*: [`step2_clean.py:L15-L35`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step2_clean.py#L15-L35)
  - **Sample Retention Filter**:
    $$\text{Retain Sample } i \iff \text{K2Q35A}_i \in \{1, 2\} \implies N = 47,768 \text{ valid samples}$$
    - *Line Reference*: [`step2_clean.py:L28-L40`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step2_clean.py#L28-L40)

---

### 1.2. `step3_inspect.py`
- **File**: [`Environmental_complete_code_directory/src/step3_inspect.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step3_inspect.py)
- **Input**: `environmental_step2_target_clean.csv`
- **Formulas & Mathematical Logic**:
  - **Column Missingness Fraction**:
    $$\text{MissingRatio}_j = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(X_{i, j} = \text{NaN}) \times 100$$
    - *Line Reference*: [`step3_inspect.py:L15-L30`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step3_inspect.py#L15-L30)

---

## 2. Preprocessing & Column Transformations (Step 4)

### 2.1. `step4_preprocess.py`
- **File**: [`Environmental_complete_code_directory/src/step4_preprocess.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step4_preprocess.py)
- **Input**: `environmental_step2_target_clean.csv`
- **Output**: `X_train_environmental.csv`, `X_test_environmental.csv`, `environmental_preprocessor.joblib`
- **Formulas & Mathematical Logic**:
  - **Sparse Column Dropping**:
    $$X_{\text{filtered}} = X \setminus \{\text{k9q41}\} \quad (\text{extremely high missingness})$$
    - *Line Reference*: [`step4_preprocess.py:L23-L28`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step4_preprocess.py#L23-L28)
  - **Stratified Train/Test Partitioning**:
    $$N_{\text{train}} = 0.80 \times 47,768 = 38,214, \quad N_{\text{test}} = 0.20 \times 47,768 = 9,554$$
    - *Line Reference*: [`step4_preprocess.py:L98-L105`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step4_preprocess.py#L98-L105)
  - **Numeric Median Imputation & Standardization**:
    $$\hat{x}_{\text{num}} = \text{Median}(x_{\text{num}}), \quad z_{\text{num}} = \frac{\hat{x}_{\text{num}} - \mu_{\text{num}}}{\sigma_{\text{num}}}$$
    - *Line Reference*: [`step4_preprocess.py:L115-L135`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step4_preprocess.py#L115-L135)
  - **Categorical One-Hot Encoding**:
    $$\mathbf{x}_{\text{cat}}^{(k)} = [ \mathbb{I}(x_{\text{cat}} = c_1), \mathbb{I}(x_{\text{cat}} = c_2), \dots, \mathbb{I}(x_{\text{cat}} = c_K) ]^T$$
    - *Line Reference*: [`step4_preprocess.py:L130-L150`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step4_preprocess.py#L130-L150)

---

## 3. Feature Selection & Consensus Importance (Steps 5–8)

### 3.1. `step5_feature_selection.py` & `step5b_group_features.py`
- **Files**: [`step5_feature_selection.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step5_feature_selection.py), [`step5b_group_features.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step5b_group_features.py)
- **Formulas & Mathematical Logic**:
  - **Mutual Information Non-Parametric Density Score**:
    $$I(X_j; Y) = \iint p(x_j, y) \log \frac{p(x_j, y)}{p(x_j) p(y)} \, dx_j \, dy$$
    - *Line Reference*: [`step5_feature_selection.py:L29-L34`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step5_feature_selection.py#L29-L34)
  - **Grouped Mutual Information Summation**:
    $$I_{\text{group}}(G) = \sum_{j \in G} I(X_j; Y)$$
    - *Line Reference*: [`step5b_group_features.py:L35-L55`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step5b_group_features.py#L35-L55)

---

### 3.2. `step6_model_feature_selection.py` & `step6b_group_rf.py`
- **Files**: [`step6_model_feature_selection.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step6_model_feature_selection.py), [`step6b_group_rf.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step6b_group_rf.py)
- **Formulas & Mathematical Logic**:
  - **Random Forest Gini Impurity Reduction**:
    $$\text{Gini}(T) = 1 - \sum_{k=0}^1 p_k^2, \quad \Delta \text{Gini}(X_j) = \text{Gini}(T_{\text{parent}}) - \sum_{s \in \{\text{left}, \text{right}\}} \frac{N_s}{N} \text{Gini}(T_s)$$
    - *Line Reference*: [`step6_model_feature_selection.py:L30-L50`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step6_model_feature_selection.py#L30-L50)

---

### 3.3. `step7_select_features.py` & `step8_prepare_selected.py`
- **Files**: [`step7_select_features.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step7_select_features.py), [`step8_prepare_selected.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step8_prepare_selected.py)
- **Formulas & Mathematical Logic**:
  - **Consensus Feature Selection**:
    $$\text{Selected Features} = \{X_j \mid \text{Rank}_{\text{MI}}(X_j) \le R_{\text{thresh}} \land \text{Rank}_{\text{RF}}(X_j) \le R_{\text{thresh}}\} \implies 31 \text{ variables}$$
    - *Line Reference*: [`step7_select_features.py:L25-L45`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step7_select_features.py#L25-L45)

---

## 4. Deep Representation Learning (Step 9)

### 4.1. `step9_environmental_representation.py`
- **File**: [`Environmental_complete_code_directory/src/step9_environmental_representation.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py)
- **Output**: `environmental_representation_model.keras`, `environmental_encoder_128d.pt`, `environmental_embedding_train.csv` ($47,768 \times 128$)
- **Formulas & Mathematical Logic**:
  - **Balanced Class Weighting**:
    $$w_c = \frac{N_{\text{total}}}{2 \cdot N_c} \implies w_0 = \frac{38214}{2 \times 36725} = 0.520, \quad w_1 = \frac{38214}{2 \times 1489} = 11.852$$
    - *Line Reference*: [`step9_environmental_representation.py:L70-L84`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py#L70-L84)
  - **Layer 1 Transformation**:
    $$\mathbf{h}_1 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_1 \mathbf{x} + b_1 \right) \right) \in \mathbb{R}^{128}, \quad \mathbf{h}_{1, \text{drop}} = \text{Dropout}_{0.20}(\mathbf{h}_1)$$
    - *Line Reference*: [`step9_environmental_representation.py:L96-L104`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py#L96-L104)
  - **Layer 2 Transformation**:
    $$\mathbf{h}_2 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_2 \mathbf{h}_{1, \text{drop}} + b_2 \right) \right) \in \mathbb{R}^{128}, \quad \mathbf{h}_{2, \text{drop}} = \text{Dropout}_{0.20}(\mathbf{h}_2)$$
    - *Line Reference*: [`step9_environmental_representation.py:L106-L114`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py#L106-L114)
  - **Embedding Representation Vector**:
    $$\mathbf{z}_{\text{env}} = W_{\text{emb}} \mathbf{h}_{2, \text{drop}} + b_{\text{emb}} \in \mathbb{R}^{128}$$
    - *Line Reference*: [`step9_environmental_representation.py:L121-L126`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py#L121-L126)
  - **Classifier Output**:
    $$\hat{y} = \sigma\left( W_{\text{cls}} \mathbf{z}_{\text{env}} + b_{\text{cls}} \right) = \frac{1}{1 + e^{-(W_{\text{cls}} \mathbf{z}_{\text{env}} + b_{\text{cls}})}}$$
    - *Line Reference*: [`step9_environmental_representation.py:L132-L137`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py#L132-L137)
  - **Class-Weighted Binary Cross-Entropy Loss**:
    $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ 11.852 \cdot y_i \log \hat{y}_i + 0.520 \cdot (1 - y_i) \log (1 - \hat{y}_i) \right]$$
    - *Line Reference*: [`step9_environmental_representation.py:L150-L165`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step9_environmental_representation.py#L150-L165)

---

## 5. Decision Threshold Optimization & Embedding Validation (Steps 10–11)

### 5.1. `step10_threshold_analysis.py` & `step11_validate_embeddings.py`
- **Files**: [`step10_threshold_analysis.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step10_threshold_analysis.py), [`step11_validate_embeddings.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step11_validate_embeddings.py)
- **Formulas & Mathematical Logic**:
  - **F1-Score Maximization Threshold Search**:
    $$t^* = \arg\max_{t \in (0, 1)} F_1(t) = \arg\max_{t \in (0, 1)} \frac{2 \cdot \text{Precision}(t) \cdot \text{Recall}(t)}{\text{Precision}(t) + \text{Recall}(t)}$$
    - *Line Reference*: [`step10_threshold_analysis.py:L35-L60`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step10_threshold_analysis.py#L35-L60)
  - **Embedding Integrity Verification**:
    $$\sum_{i=1}^N \sum_{j=1}^{128} \mathbb{I}(\mathbf{z}_{i, j} = \text{NaN}) = 0, \quad \sum_{i=1}^N \sum_{j=1}^{128} \mathbb{I}(\mathbf{z}_{i, j} = \pm \infty) = 0$$
    - *Line Reference*: [`step11_validate_embeddings.py:L40-L55`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step11_validate_embeddings.py#L40-L55)

---

## 6. Cross-Modal 32-D Autoencoder Alignment (Steps 12–15)

### 6.1. `step12_reduce_embedding.py` & `step13_validate_32d.py`
- **Files**: [`step12_reduce_embedding.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step12_reduce_embedding.py), [`step13_validate_32d.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step13_validate_32d.py)
- **Output**: `environmental_128_to_32_alignment.pt`, `environmental_32d_aligned_representation.csv` ($47,768 \times 32$)
- **Formulas & Mathematical Logic**:
  - **Encoder Subnetwork**:
    $$\mathbf{h}_1 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_1 \mathbf{x}_{128} + b_1 \right) \right) \in \mathbb{R}^{64}$$
    $$\mathbf{x}_{32} = W_2 \mathbf{h}_1 + b_2 \in \mathbb{R}^{32}$$
    - *Line Reference*: [`step12_reduce_embedding.py:L45-L60`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step12_reduce_embedding.py#L45-L60)
  - **Decoder Subnetwork**:
    $$\mathbf{h}_2 = \text{ReLU}\left( \text{BatchNorm1d}\left( W_3 \mathbf{x}_{32} + b_3 \right) \right) \in \mathbb{R}^{64}$$
    $$\hat{\mathbf{x}}_{128} = W_4 \mathbf{h}_2 + b_4 \in \mathbb{R}^{128}$$
    - *Line Reference*: [`step12_reduce_embedding.py:L62-L75`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step12_reduce_embedding.py#L62-L75)
  - **Autoencoder Reconstruction MSE Loss**:
    $$\mathcal{L}_{\text{recon}} = \frac{1}{128 N} \sum_{i=1}^N \|\mathbf{x}_{128, i} - \hat{\mathbf{x}}_{128, i}\|_2^2 = 0.135754$$
    - *Line Reference*: [`step12_reduce_embedding.py:L80-L95`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step12_reduce_embedding.py#L80-L95)

---

### 6.2. `step14_add_target_to_32d.py` & `step15_evaluate_32d.py`
- **Files**: [`step14_add_target_to_32d.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step14_add_target_to_32d.py), [`step15_evaluate_32d.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step15_evaluate_32d.py)
- **Formulas & Mathematical Logic**:
  - **ROC-AUC Integral**:
    $$\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt = 0.6729$$
    - *Line Reference*: [`step15_evaluate_32d.py:L110-L135`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/Environmental_complete_code_directory/src/step15_evaluate_32d.py#L110-L135)

---

## 7. Production Backend Service Integration

### 7.1. `backend/services/environmental_service.py`
- **File**: [`backend/services/environmental_service.py`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py)
- **Formulas & Mathematical Logic**:
  - **Single-Patient Preprocessing Transformation**:
    $$\mathbf{x}_{31} = \text{ColumnTransformer}(\mathbf{x}_{\text{raw}})$$
    - *Line Reference*: [`environmental_service.py:L260-L280`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py#L260-L280)
  - **128-D Encoder Forward Pass**:
    $$\mathbf{x}_{128} = \text{EnvironmentalEncoder.encode}(\mathbf{x}_{31})$$
    - *Line Reference*: [`environmental_service.py:L129-L138`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py#L129-L138)
  - **32-D Aligned Forward Pass**:
    $$\mathbf{x}_{32} = \text{EnvironmentalAlignment.forward}(\mathbf{x}_{128})$$
    - *Line Reference*: [`environmental_service.py:L192-L200`](file:///d:/GAN-Enhanced-Multimodal-ASD-Screening/backend/services/environmental_service.py#L192-L200)
