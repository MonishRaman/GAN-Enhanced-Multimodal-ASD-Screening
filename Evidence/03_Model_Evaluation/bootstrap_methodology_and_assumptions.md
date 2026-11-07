# Statistical Validation Methodology & Dependence Structure Audit

**Audit Date:** 2026-10-30 16:38:46  
**Evaluation Target:** 5,000 In-Silico Multimodal Test Combinations  
**Checkpoint:** `08_hierarchical_minn/models/minn_best.pt`  

## 1. The Multi-Way Crossed Clustering Structure
The 5,000 in-silico test combinations are formed by drawing class-conditionally with replacement across three independent source databases:
- **Genetic (GSE28521):** 24 held-out subjects (12 ASD, 12 Control). Reused ~208.33× across the 5,000 combinations.
- **Behavioral (Toddler Q-CHAT-10):** 132 held-out records (66 ASD, 66 Control). Reused ~37.88×.
- **Environmental (NSCH 2023):** 7,166 held-out records (318 ASD, 6,848 Control). Reused ~7.86× for ASD and 0.37× for Control.

Each combination $i \in \{1, \dots, 5000\}$ is indexed as:
$$T_i = (G_{g(i)}, B_{b(i)}, E_{e(i)}), \quad g(i) \in \{1 \dots 24\}, \; b(i) \in \{1 \dots 132\}, \; e(i) \in \{1 \dots 7166\}$$

Because each modality is sampled with replacement, rows share underlying biological, behavioral, and environmental records. Consequently, the standard statistical assumption of independent and identically distributed (i.i.d.) observations is violated.

## 2. Comparison of Resampling Methods (1,000 Iterations)

| Metric | Point Estimate | Method 1: Naive Row Bootstrap (i.i.d.) | Method 2: Genetic Cluster Bootstrap (PSU=24) | Method 3: Two-Way Crossed (G x B) | Method 4: Three-Way Crossed (G x B x E) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.9454 | **[0.9392, 0.9518]** (w=0.0126) | **[0.9166, 0.9699]** (w=0.0533) | **[0.9131, 0.9743]** (w=0.0259) | **[0.9146, 0.9758]** (w=0.0266) |
| **Sensitivity** | 0.9652 | **[0.9576, 0.9717]** (w=0.0141) | **[0.9287, 0.9909]** (w=0.0622) | **[0.9362, 0.9924]** (w=0.0562) | **[0.9347, 0.9935]** (w=0.0588) |
| **Specificity** | 0.9256 | **[0.9153, 0.9362]** (w=0.0208) | **[0.8792, 0.9676]** (w=0.0884) | **[0.8671, 0.9724]** (w=0.1052) | **[0.8720, 0.9758]** (w=0.1038) |
| **ROC-AUC** | 0.9876 | **[0.9848, 0.9901]** (w=0.0053) | **[0.9670, 0.9977]** (w=0.0307) | **[0.9723, 0.9982]** (w=0.0259) | **[0.9717, 0.9983]** (w=0.0266) |

## 3. Methodological Assumptions & Statistical Verdict
1. **Method 1 (Naive Row Bootstrap):**  
   *Assumption:* Assumes 5,000 independent patients.  
   *Verdict:* **FLAWED / OVERCONFIDENT.** Repeating 24 genetic subjects ~208× while treating each row as independent artificially deflates standard errors by a factor of 4x.
2. **Method 2 (Genetic Primary Cluster Bootstrap):**  
   *Assumption:* Resamples the 24 genetic subjects as clusters.  
   *Verdict:* **STATISTICALLY DEFENSIBLE FOR GENETIC BOTTLENECK.** Because $N_G = 24$ is the smallest finite sample, genetic variance dominates the resampling uncertainty.
3. **Method 3 & 4 (Two-Way and Three-Way Crossed Cluster Bootstrap):**  
   *Assumption:* Resamples genetic, behavioral, and environmental source records simultaneously as crossed clusters.  
   *Verdict:* **RIGOROUS MULTI-WAY DESIGN ESTIMATE.** Confirms that accounting for crossed clustering yields confidence interval widths (~0.05-0.06) consistent with Method 2, demonstrating that the genetic bottleneck is indeed the primary driver of finite-sample uncertainty.

## 4. Fundamental Epidemiological Scope Limitation
> **Mandatory Scientific Disclaimer:**  
> The estimand of these confidence intervals is the expected test performance over **in-silico combinations drawn from the Cartesian product of these three specific retrospective cohorts**.  
> **These intervals do NOT estimate population-level clinical screening performance for live, prospectively recruited pediatric patients.**  
> Because genetic microarrays were extracted from post-mortem cerebral cortex tissue, behavioral data from toddler screening questionnaires, and environmental data from NSCH survey matrices, the data cannot establish real-world biological correlation structures or prove clinical generalizability.
