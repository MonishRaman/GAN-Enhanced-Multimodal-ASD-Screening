# Complete & Exhaustive Mathematical Analysis of Genetics Code Files

---

## 1. Data Ingestion & Preprocessing (Steps 1–9)

### 1.1. `step1_merge_samples.py`
- **Location**: `Genetics_Complete_code/src/step1_merge_samples.py`
- **Description**: Merges $N=116$ raw microarray GSM sample tables into a single expression matrix $E \in \mathbb{R}^{P \times N}$ using the probe ID (`Position`) and `Channel_A` signal intensity.
- **Formulas**:
  - **Sample Concatenation**:
    $$E_{i, j} = \text{Signal}_{Channel\_A}(\text{Probe}_i, \text{GSM}_j), \quad i \in \{1, \dots, P\}, \ j \in \{1, \dots, 116\}$$
    - *Line Reference*: `step1_merge_samples.py:L25-L42`

---

### 1.2. `step2_data_quality_assessment.py` & `step2_5_missing_distribution.py`
- **Location**: `step2_data_quality_assessment.py`, `step2_5_missing_distribution.py`
- **Description**: Computes overall sample-wise and probe-wise missing value ratios.
- **Formulas**:
  - **Probe-wise Missing Fraction**:
    $$M_i = \frac{1}{N} \sum_{j=1}^N \mathbb{I}(E_{i, j} = \text{NaN})$$
    - *Line Reference*: `step2_data_quality_assessment.py:L35-L45`
  - **Sample-wise Missing Fraction**:
    $$S_j = \frac{1}{P} \sum_{i=1}^P \mathbb{I}(E_{i, j} = \text{NaN})$$
    - *Line Reference*: `step2_5_missing_distribution.py:L40-L52`

---

### 1.3. `step3_filter_probes.py`
- **Location**: `Genetics_Complete_code/src/step3_filter_probes.py`
- **Description**: Filters out noisy probes exceeding $75\%$ missingness across all 116 samples.
- **Formulas**:
  - **Probe Filtering Rule**:
    $$\text{MissingRatio}_i = \frac{\sum_{j=1}^N \mathbb{I}(E_{i, j} = \text{NaN})}{N} \times 100$$
    $$\text{Keep Probe } i \iff \text{MissingRatio}_i \le 75\%$$
    - *Line Reference*: `step3_filter_probes.py:L41-L65`

---

### 1.4. `step4_knn_imputation.py`
- **Location**: `Genetics_Complete_code/src/step4_knn_imputation.py`
- **Description**: Imputes missing expression intensities using $k$-Nearest Neighbors ($k=5$, uniform weighting).
- **Formulas**:
  - **Euclidean Distance Between Samples**:
    $$d(u, v) = \sqrt{\sum_{m \in O_u \cap O_v} (E_{m, u} - E_{m, v})^2}$$
    - *Line Reference*: `step4_knn_imputation.py:L46-L51`
  - **Uniform $k$-NN Imputation**:
    $$\hat{E}_{i, u} = \frac{1}{k} \sum_{v \in N_k(u)} E_{i, v} \quad (k=5)$$
    - *Line Reference*: `step4_knn_imputation.py:L51-L56`

---

### 1.5. `step5_zscore_normalization.py`
- **Location**: `Genetics_Complete_code/src/step5_zscore_normalization.py`
- **Description**: Standardizes each probe row across samples to zero mean ($\mu_i = 0$) and unit variance ($\sigma_i = 1$).
- **Formulas**:
  - **Probe Mean & Standard Deviation**:
    $$\mu_i = \frac{1}{N} \sum_{j=1}^N E_{i, j}, \quad \sigma_i = \sqrt{\frac{1}{N} \sum_{j=1}^N (E_{i, j} - \mu_i)^2}$$
    - *Line Reference*: `step5_zscore_normalization.py:L46-L49`
  - **Z-Score Transformation**:
    $$Z_{i, j} = \frac{E_{i, j} - \mu_i}{\sigma_i}$$
    - *Line Reference*: `step5_zscore_normalization.py:L48`

---

### 1.6. `step6_extract_labels.py` to `step9_dataset_validation.py`
- **Location**: `step6_extract_labels.py` through `step9_dataset_validation.py`
- **Description**: Maps GSM sample IDs to phenotypic targets ($y \in \{0, 1\}$) and constructs transposed ML feature matrix $X \in \mathbb{R}^{116 \times P}$.
- **Formulas**:
  - **Phenotypic Target Mapping**:
    $$y_j = \begin{cases} 1 & \text{if Target = ASD} \\ 0 & \text{if Target = Control} \end{cases}$$
    - *Line Reference*: `step6_extract_labels.py:L30-L50`
  - **Class Ratio Calculation**:
    $$\text{Ratio}_{\text{Control}} = \frac{N_{\text{Control}}}{N_{\text{total}}} = \frac{87}{116} = 0.75, \quad \text{Ratio}_{\text{ASD}} = \frac{29}{116} = 0.25$$
    - *Line Reference*: `step9_dataset_validation.py:L40-L60`

---

## 2. Feature Selection & Filtering (Steps 10–12)

### 2.1. `step10_variance_threshold.py`
- **Location**: `Genetics_Complete_code/src/step10_variance_threshold.py`
- **Description**: Removes uninformative probes with variance below $0.01$.
- **Formulas**:
  - **Sample Variance**:
    $$\text{Var}(X_i) = \frac{1}{N} \sum_{j=1}^N (X_{j, i} - \bar{X}_i)^2$$
    $$\text{Keep Feature } i \iff \text{Var}(X_i) \ge 0.01$$
    - *Line Reference*: `step10_variance_threshold.py:L41-L47`

---

### 2.2. `step11_mutual_information.py`
- **Location**: `Genetics_Complete_code/src/step11_mutual_information.py`
- **Description**: Ranks probes by Mutual Information with binary target labels using $k$-NN density estimation to select the **Top 1000 features**.
- **Formulas**:
  - **Mutual Information**:
    $$I(X_i; Y) = \iint p(x_i, y) \log \frac{p(x_i, y)}{p(x_i) p(y)} \, dx_i \, dy$$
    - *Line Reference*: `step11_mutual_information.py:L47-L56`
  - **$k$-NN Density Estimation (Kraskov Estimator)**:
    $$I(X_i; Y) = \psi(N) - \langle \psi(N_y) \rangle + \psi(k) - \langle \psi(m_i) \rangle$$
    - *Line Reference*: `step11_mutual_information.py:L47-L51`

---

### 2.3. `step12_correlation_filter.py`
- **Location**: `Genetics_Complete_code/src/step12_correlation_filter.py`
- **Description**: Computes upper-triangular Pearson correlation matrix and drops redundant features with $|r| > 0.95$.
- **Formulas**:
  - **Pearson Correlation Coefficient**:
    $$r(X_a, X_b) = \frac{\sum_{j=1}^N (X_{j, a} - \bar{X}_a)(X_{j, b} - \bar{X}_b)}{\sqrt{\sum_{j=1}^N (X_{j, a} - \bar{X}_a)^2 \sum_{j=1}^N (X_{j, b} - \bar{X}_b)^2}}$$
    $$\text{Drop } X_b \iff \exists \, a < b \text{ s.t. } |r(X_a, X_b)| > 0.95$$
    - *Line Reference*: `step12_correlation_filter.py:L40-L50`

---

## 3. Representation Learning & Transformer Architectures

### 3.1. `Genetics_Complete_code/genetics/model.py` (`GeneticTransformer`)
- **Location**: `Genetics_Complete_code/genetics/model.py`
- **Description**: Standard PyTorch Genetic Transformer model with scalar token projections and global average pooling.
- **Formulas**:
  - **Scalar Feature Linear Projection**:
    $$\mathbf{h}_i^{(0)} = W_{\text{proj}} x_i + b_{\text{proj}} \in \mathbb{R}^{d}, \quad x_i \in \mathbb{R}^1 \quad (d=64)$$
    - *Line Reference*: `model.py:L28-L31`, `L95-L99`
  - **Learnable Positional Encoding Addition**:
    $$\mathbf{E}_i = \mathbf{h}_i^{(0)} + \mathbf{P}_i, \quad \mathbf{P} \sim \mathcal{N}(0, I) \in \mathbb{R}^{1 \times 1000 \times d}$$
    - *Line Reference*: `model.py:L37-L43`, `L103`
  - **Scaled Multi-Head Self-Attention**:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V, \quad d_k = \frac{d}{h} = \frac{64}{4} = 16$$
    - *Line Reference*: `model.py:L49-L62`, `L107`
  - **GELU Activation**:
    $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$
    - *Line Reference*: `model.py:L54`
  - **Layer Normalization**:
    $$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
    - *Line Reference*: `model.py:L68-L70`, `L111`
  - **Global Average Pooling**:
    $$\mathbf{z}_{\text{embedding}} = \frac{1}{1000} \sum_{i=1}^{1000} \mathbf{h}_i^{(\text{final})} \in \mathbb{R}^{64}$$
    - *Line Reference*: `model.py:L115`
  - **Linear Classifier Logits**:
    $$\hat{\mathbf{y}} = W_{\text{cls}} \cdot \text{Dropout}(\mathbf{z}_{\text{embedding}}) + b_{\text{cls}} \in \mathbb{R}^2$$
    - *Line Reference*: `model.py:L76-L84`, `L129-L131`

---

### 3.2. `step28_train_genetic_transformer.py` (`LightweightGeneticTransformer`)
- **Location**: `Genetics_Complete_code/src/step28_train_genetic_transformer.py`
- **Description**: Primary production Transformer model compressing 1000 probes into 32 tokens of 128 dimensions with attention-weighted pooling.
- **Formulas**:
  - **Token Compression Projection**:
    $$Y = X \cdot W_{\text{proj}}^T + b_{\text{proj}} \in \mathbb{R}^{B \times 4096} \implies H \in \mathbb{R}^{B \times 32 \times 128}$$
    - *Line Reference*: `step28_train_genetic_transformer.py:L149-L152`, `L199-L205`
  - **Positional Tensor Initializer**:
    $$\mathbf{P} \sim \mathcal{N}(0, 0.02^2 \cdot I) \in \mathbb{R}^{1 \times 32 \times 128}$$
    - *Line Reference*: `step28_train_genetic_transformer.py:L155-L161`
  - **Learned Attention-Weighted Token Pooling**:
    $$s_i = W_{\text{att}} h_i + b_{\text{att}} \in \mathbb{R}^1, \quad \alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{32} \exp(s_j)}$$
    $$\mathbf{z}_{\text{pooled}} = \sum_{i=1}^{32} \alpha_i \cdot h_i \in \mathbb{R}^{128}$$
    - *Line Reference*: `step28_train_genetic_transformer.py:L178-L181`, `L222-L232`
  - **Class Weight Calculation**:
    $$w_c = \frac{N_{\text{total}}}{2 \cdot N_c} \implies w_{\text{Control}} = \frac{116}{2 \cdot 87} = 0.667, \quad w_{\text{ASD}} = \frac{116}{2 \cdot 29} = 2.000$$
    *(Empirically calibrated to $w_{\text{Control}}=1.33, w_{\text{ASD}}=4.00$)*
    - *Line Reference*: `step28_train_genetic_transformer.py:L293-L300`
  - **Weighted Cross-Entropy Loss**:
    $$\mathcal{L}_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^N w_{y_i} \log \left( \frac{\exp(\hat{y}_{i, y_i})}{\sum_{j=0}^1 \exp(\hat{y}_{i, j})} \right)$$
    - *Line Reference*: `step28_train_genetic_transformer.py:L305-L310`

---

## 4. Generative Data Augmentation & Synthetic Validation (GANs & CTGAN)

### 4.1. `step31_genetic_conditional_gan.py`
- **Location**: `Genetics_Complete_code/src/step31_genetic_conditional_gan.py`
- **Description**: Conditional GAN generating 46 synthetic control samples directly in 1000-feature gene expression space.
- **Formulas**:
  - **Generator Mapping**:
    $$G(z, y) = W_3 \cdot \text{ReLU}\left( \text{BatchNorm}\left( W_2 \cdot \text{ReLU}\left( \text{BatchNorm}\left( W_1 [z \parallel E_y] \right) \right) \right) \right)$$
    $$\text{where } z \sim \mathcal{N}(0, I_{128}), \ E_y \in \mathbb{R}^{32}, \ G(z,y) \in \mathbb{R}^{1000}$$
    - *Line Reference*: `step31_genetic_conditional_gan.py:L269-L323`
  - **Discriminator Mapping**:
    $$D(x, y) = W_3 \cdot \text{LeakyReLU}_{0.2}\left( \text{Dropout}_{0.2}\left( W_2 \cdot \text{LeakyReLU}_{0.2}\left( \text{Dropout}_{0.2}\left( W_1 [x \parallel E_y] \right) \right) \right) \right)$$
    - *Line Reference*: `step31_genetic_conditional_gan.py:L329-L384`
  - **Conditional Minimax Objective**:
    $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x \mid y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z \mid y) \mid y))]$$
    - *Line Reference*: `step31_genetic_conditional_gan.py:L407-L565`

---

### 4.2. `step32_genetic_synthetic_validation.py` & `step34_genetic_embedding_synthetic_validation.py`
- **Location**: `step32_genetic_synthetic_validation.py`, `step34_genetic_embedding_synthetic_validation.py`
- **Description**: Statistical validation of synthetic samples against real distributions.
- **Formulas**:
  - **Kolmogorov-Smirnov (KS) 2-Sample Test Statistic**:
    $$D_k = \sup_x |F_{\text{real}, k}(x) - F_{\text{synth}, k}(x)|, \quad k \in \{1, \dots, 1000\}$$
    - *Line Reference*: `step32_genetic_synthetic_validation.py:L158-L165`
  - **Euclidean Distance to Nearest Real Neighbor**:
    $$d_{\text{nearest}}(s_i) = \min_{r_j \in \text{Real}} \|s_i - r_j\|_2 = \min_{r_j} \sqrt{\sum_{k=1}^{1000} (s_{i,k} - r_{j,k})^2}$$
    - *Line Reference*: `step32_genetic_synthetic_validation.py:L346-L360`
  - **Pairwise Synthetic Diversity Metric**:
    $$\text{Diversity} = \frac{2}{M(M-1)} \sum_{1 \le i < j \le M} \|s_i - s_j\|_2$$
    - *Line Reference*: `step32_genetic_synthetic_validation.py:L370-L388`
  - **2-D PCA Projection Matrix**:
    $$X_{\text{PCA}} = X_{\text{concat}} \cdot V_2, \quad \text{where } \Sigma = V \Lambda V^T \in \mathbb{R}^{1000 \times 1000}$$
    - *Line Reference*: `step32_genetic_synthetic_validation.py:L210-L245`

---

### 4.3. `step33_genetic_embedding_conditional_gan.py` & `genetic_ctgan.py`
- **Location**: `step33_genetic_embedding_conditional_gan.py`, `genetic_ctgan.py`
- **Description**: Synthesizes 128-D latent transformer embeddings using CGAN and SDV `CTGANSynthesizer` (VGM Gaussian Mixture representation).
- **Formulas**:
  - **Variational Gaussian Mixture (VGM) Normalization**:
    $$c_{i, j} \sim \sum_{k=1}^K \pi_{i, k} \mathcal{N}(\mu_{i, k}, \sigma_{i, k}^2)$$
    - *Line Reference*: `genetic_ctgan.py:L85-L120`
  - **Wasserstein GAN with Gradient Penalty (WGAN-GP) Loss**:
    $$\mathcal{L}_D = \mathbb{E}[\tilde{D}(\hat{x})] - \mathbb{E}[\tilde{D}(x)] + \lambda \mathbb{E}_{\hat{x}}\left[ (\|\nabla_{\hat{x}} \tilde{D}(\hat{x})\|_2 - 1)^2 \right]$$
    - *Line Reference*: `genetic_ctgan.py:L180-L230`

---

### 4.4. `step35_genetic_gan_utility_validation.py` to `step37_fold_safe_genetic_gan_validation.py`
- **Location**: `step35_genetic_gan_utility_validation.py` through `step37_fold_safe_genetic_gan_validation.py`
- **Description**: Downstream sensitivity evaluation across synthetic sample counts ($K \in \{0, 10, 20, 30, 46\}$) and fold-isolated leakage-free GAN training.
- **Formulas**:
  - **Balanced Accuracy**:
    $$\text{Balanced Acc} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$$
    - *Line Reference*: `step35_genetic_gan_utility_validation.py:L140-L155`
  - **Precision & Recall**:
    $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
    - *Line Reference*: `step35_genetic_gan_utility_validation.py:L156-L162`
  - **F1-Score**:
    $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$
    - *Line Reference*: `step35_genetic_gan_utility_validation.py:L163-L168`

---

## 5. Cross-Modal 32-D Alignment & Production Backend Services

### 5.1. `step38_genetic_128_to_32_alignment.py` & `step39_validate_genetic_32d.py`
- **Location**: `step38_genetic_128_to_32_alignment.py`, `step39_validate_genetic_32d.py`
- **Description**: Projects 128-D latent embeddings into a compact 32-D aligned vector space for multimodal evidence fusion.
- **Formulas**:
  - **Alignment Projection Network**:
    $$\mathbf{x}_{32} = \text{Linear}_{64 \to 32}\left( \text{ReLU}\left( \text{BatchNorm1d}\left( \text{Linear}_{128 \to 64}(\mathbf{x}_{128}) \right) \right) \right)$$
    - *Line Reference*: `step38_genetic_128_to_32_alignment.py:L122-L136`

---

### 5.2. `backend/services/genetic_service.py`
- **Location**: `backend/services/genetic_service.py`
- **Description**: Production FastAPI inference handler processing raw single-patient 1000-feature inputs into 32-D aligned representations.
- **Formulas**:
  - **Sequence Feature Embedding & Transformer Encoding**:
    $$\mathbf{h}_i = \text{Linear}_{1 \to 128}(x_i), \quad H_{\text{trans}} = \text{TransformerEncoder}(\mathbf{h}_1, \dots, \mathbf{h}_{1000})$$
    - *Line Reference*: `genetic_service.py:L84-L97`, `L111-L117`
  - **Mean Feature Representation Aggregation**:
    $$\mathbf{x}_{128} = \text{LayerNorm}\left( \text{Linear}_{128 \to 128}\left( \frac{1}{1000} \sum_{i=1}^{1000} H_{\text{trans}, i} \right) \right)$$
    - *Line Reference*: `genetic_service.py:L120-L125`
  - **32-D Aligned Latent Vector Output**:
    $$\mathbf{x}_{32} = \text{Linear}_{64 \to 32}\left( \text{ReLU}\left( \text{BatchNorm1d}\left( \text{Linear}_{128 \to 64}(\mathbf{x}_{128}) \right) \right) \right)$$
    - *Line Reference*: `genetic_service.py:L148-L160`

---

### 5.3. `backend/services/fusion_service.py` (Genetic Expert Branch)
- **Location**: `backend/services/fusion_service.py`
- **Description**: Standardizes 32-D genetic aligned inputs, evaluates the Genetic Expert MLP, and combines probabilities using weighted evidence fusion.
- **Formulas**:
  - **Genetic 32-D Input Scaling**:
    $$\mathbf{z}_{\text{gen}} = \frac{\mathbf{x}_{\text{gen}, 32} - \boldsymbol{\mu}_{\text{gen}}}{\boldsymbol{\sigma}_{\text{gen}}}$$
    - *Line Reference*: `fusion_service.py:L310-L345`
  - **Genetic Expert MLP Probability**:
    $$P_{\text{Genetic}} = \sigma \left( W_3 \cdot \text{ReLU}\left( W_2 \cdot \text{Dropout}_{0.2}\left( \text{ReLU}\left( \text{BatchNorm1d}\left( W_1 \mathbf{z}_{\text{gen}} + b_1 \right) \right) \right) + b_2 \right) + b_3 \right)$$
    - *Line Reference*: `fusion_service.py:L63-L101`, `L340-L355`
  - **Multimodal Weighted Evidence Fusion**:
    $$P_{\text{fused}} = 0.319370 \cdot P_{\text{Genetic}} + 0.400219 \cdot P_{\text{Behavioral}} + 0.280411 \cdot P_{\text{Environmental}}$$
    - *Line Reference*: `fusion_service.py:L50-L54`, `L356-L365`

---

## Summary Checklist of Genetics Equations & References

- [x] **Data Ingestion & Preprocessing**: Documented Probe Filtering Ratio, Euclidean Distance, Uniform $k$-NN Imputation, and Probe-wise Z-Score Standardization.
- [x] **Feature Selection**: Documented Variance Threshold ($\ge 0.01$), Mutual Information ($k$-NN Kraskov estimator), and Upper-Triangular Pearson Correlation ($|r| \le 0.95$).
- [x] **Transformer Architectures**: Documented Token Compression Linear Projection ($1000 \to 32 \times 128$), Learnable Positional Tensor Addition, Multi-Head Self-Attention ($h=4, d_k=32$), Attention-Weighted Softmax Pooling, and Weighted Cross-Entropy Loss.
- [x] **Generative Data Augmentation**: Documented Conditional GAN Minimax Objective, Generator/Discriminator Architectures, Kolmogorov-Smirnov 2-Sample Test Statistic $D$, Nearest Neighbor Euclidean Distance $d_{\text{nearest}}$, Pairwise Diversity Metric, and WGAN-GP Loss.
- [x] **Alignment & Production Backend**: Documented 128-D $\to$ 32-D Linear-BatchNorm-ReLU Alignment Projection, Genetic Expert MLP Evaluation, and Weighted Multimodal Evidence Fusion.