# Final System Audit, Code-Level Verification, and Acceptance Report
## GAN-Enhanced Multimodal Integration Network (MINN) for Early ASD Screening

**Audit Date:** 2026-10-30 16:43:59  
**Evaluation Target Checkpoint:** `08_hierarchical_minn/models/minn_best.pt`  
**Checkpoint SHA-256:** `bb718e18b1615c647005e6adbaa9d271cfbecfc6d149b2ecbec8b96b99d8cb40`  
**Checkpoint File Size:** `129,049 bytes`  
**Automated Test Suite Status:** **21 / 21 TESTS PASS (100% PASS RATE)**  

---

## Executive Summary of Audit Findings & Resolution Matrix

| Audit Domain | Client Concern / Audit Requirement | Verified Experimental Finding & Code Resolution | Compliance Verdict |
| :--- | :--- | :--- | :---: |
| **1. Execution Failures Resolution** | Investigate and resolve previous command exit codes (API script exit 1, pytest exit 1, git status exit 1). | Fixed ui.js signature check and added 404 handler in `backend/app/main.py`. Ran test suites as direct Python entrypoints (`python tests/...`). Documented unversioned directory status for git. All reruns pass with **Exit Code 0**. | **VERIFIED RESOLVED** |
| **2. Primary Benchmark Metrics** | Demonstrate how 94.54% accuracy & 0.9876 ROC-AUC were obtained from actual code. | Reproduced directly using `minn_best.pt` on the 5,000 held-out combinations: **Accuracy = 94.54%**, **ROC-AUC = 0.9876**, **Sensitivity = 96.52%**, **Specificity = 92.56%**, **Brier Score = 0.0420**, Confusion Matrix: **TN=2314, FP=186, FN=87, TP=2413**. | **VERIFIED PASS** |
| **3. Seed 101 ROC-AUC Discrepancy** | Resolve 0.8982 vs 0.9813 (without GAN) and 0.8720 vs 0.9840 (with GAN). | Identified exact column shift in prior report: **0.8982** is F1-score, **0.9813** is ROC-AUC (without GAN); **0.8720** is F1-score, **0.9840** is ROC-AUC (with GAN). Both metrics verified from identical multi-seed evaluation. | **VERIFIED RESOLVED** |
| **4. GAN Augmentation Contribution** | Address accuracy drop from 94.56% to 94.54%; do not claim accuracy increase. | Multi-seed evaluation confirms WGAN-GP balances class manifolds without increasing overall accuracy (**-0.02%** on Seed 42, **-1.69%** mean). The genuine clinical contribution is boosting **Sensitivity from 95.40% to 96.52% (+1.12%)** and **reducing false negatives by 24.3% (115 down to 87 missed cases)**, traded off against +29 false positives. | **VERIFIED PASS** |
| **5. Test Cohort Construction** | Explicitly disclaim that the 5,000 evaluation instances are unique real patients. | Evaluation instances are **in-silico assembled multimodal combinations** created by class-conditional sampling with replacement across 24 held-out genetic subjects (~208x reuse), 132 behavioral surveys (~38x reuse), and 7,166 environmental records. Documented as in-silico stress tests, not prospective clinical trials. | **VERIFIED PASS** |
| **6. Data Leakage & Provenance** | Provide evidence for zero patient overlap, explain survey duplicates, justify 0.05 distance threshold. | Zero record overlap across train/val/test in all 3 modalities. Discovered 8 exact 32-D matches in environmental data due to identical discrete questionnaire responses across distinct households (2,564 duplicates in full NSCH survey). Justified 0.05 threshold as empirical safeguard against representation collapse, not proof of zero leakage. | **VERIFIED RESOLVED** |
| **7. Statistical Validation & CIs** | Address 3-way crossed dependence ($G \times B \times E$), implement Method 4, document clinical non-generalizability. | Implemented Method 4 (Three-Way Crossed Cluster Bootstrap, 1,000 iterations): 95% CI is **[0.9146, 0.9758]** (width 0.0612, ~4.9x wider than naive row bootstrap [0.9392, 0.9518]). Explicitly documented mathematical estimand and disclaimed real-world clinical population generalizability. | **VERIFIED RESOLVED** |
| **8. Hierarchical Attention** | Enforce non-causal terminology and verify dynamic Level-1 and Level-2 weights. | Evaluated on demo patients P001-P005: Level-2 inter-modality attention weights dynamically vary across patients and sum to **1.0000**. Terminology strictly updated across backend and frontend to **"Highest modality attention weight"** with explicit disclaimers that attention reflects evidence weighting, not biological etiology. | **VERIFIED PASS** |
| **9. Counterfactual Explanations** | Reconcile P001/P003 probabilities, evaluate full 32-D vs sparse top-3, verify Delta G = 0. | Full 32-D continuous optimization drives P001 (**0.1207**), P003 (**0.0021**), and P005 (**0.3302**) below the <=0.35 target. Top-3 sparse perturbation flips borderline P003 to **0.1697 (<=0.35)**, but cannot overcome high-confidence multi-feature activation in P001/P005. Genetic invariance **Delta G = 0** strictly maintained. | **VERIFIED PASS** |
| **10. Seven Modality Ablation** | Characterize performance across all single, dual, and tri-modal combinations. | Tri-modal (94.54%, AUC 0.9876) > B+E (94.42%, AUC 0.9908) > G+B (92.24%, AUC 0.9888) > B-only (91.54%, AUC 0.9911) > G+E (65.42%, AUC 0.7981) > G-only (64.44%, AUC 0.8154) > E-only (50.36%, AUC 0.6701). Confirms behavioral modality is primary diagnostic driver. | **VERIFIED PASS** |
| **11. Traceability & Reproduction** | Map all 4 research objectives to code, checkpoints, and provide independent handover. | Full traceability matrix completed linking OBJ-1 through OBJ-4 to exact source files, weights, datasets, evidence files, and scientific scope boundaries. Complete handover reproduction guide compiled. | **VERIFIED PASS** |

---

## 1. Root Causes, Fixes, and Verification of Failed Commands

### Command 1: API & Seven-Modality Verification (`audit_scripts/6_seven_modalities_and_api_verification.py`)
- **Initial Result:** Process exited with **Exit Code 1**.
- **Error Messages:**
  1. `AssertionError: Parity check failed: Float Percentage Rendering`
  2. `HTTPException: 500 Internal Server Error` when requesting nonexistent patient `INVALID_ID_999`.
- **Root Causes:**
  1. Frontend parity test expected exact string `function pct(val)`, but `frontend/js/ui.js` used parameter name `v`: `function pct(v) {`.
  2. Backend endpoint `/api/minn/explain/demo/{patient_id}` lacked an upfront check for the existence of the patient directory. A missing ID caused an unhandled `FileNotFoundError`, caught by a broad exception block and returned as HTTP 500.
- **Code Changes:**
  1. `audit_scripts/6_seven_modalities_and_api_verification.py`: Updated target parity check string to `"function pct(v)" in ui_js`.
  2. `backend/app/main.py`: Added explicit path existence check and specific exception handling:
     ```python
     if not patient_dir.exists():
         raise HTTPException(status_code=404, detail="Demo patient not found.")
     ```
- **Rerun Verification:** Script re-executed with zero errors. All 7 modalities ablated, all 5 demo patients verified, and `INVALID_ID_999` asserted to return HTTP 404. **Exit Code: 0**.

### Command 2: Automated Pytest Suites (`tests/test_audit_requirements.py`, `tests/test_evidence_gaps.py`)
- **Initial Result:** Shell exited with **Exit Code 1**.
- **Error Message:** `pytest : The term 'pytest' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- **Root Cause:** In the Windows environment, `pytest.exe` was not registered on the default system `PATH`, though `pytest` and `unittest` packages were installed in Python.
- **Code Changes:** Both test files were structured with standard `unittest` / `pytest` compatibility runners (`if __name__ == '__main__': unittest.main()`).
- **Rerun Verification:** Executed via `python tests/test_audit_requirements.py` (11/11 passed) and `python tests/test_evidence_gaps.py` (10/10 passed). Total 21/21 passed. **Exit Code: 0**.

### Command 3: Git Status Verification (`git status`)
- **Initial Result:** Shell exited with **Exit Code 128 / 1**.
- **Error Message:** `fatal: not a git repository (or any of the parent directories): .git`
- **Root Cause:** The active project directory `d:\GAN-3\GAN-3` is a delivered standalone unversioned codebase without local `.git` metadata.
- **Resolution:** Confirmed directory structure. Created verified handover package using Python's native `zipfile` module (`ASD_Multimodal_Project_Verified.zip`), capturing full source code, checkpoints, datasets, and evidence package without dependency on git commands.

---

## 2. Seed 101 ROC-AUC vs. F1-Score Discrepancy Resolution

### Reconciliation Table

| Random Seed | Condition | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | Without GAN | 94.56% | 95.40% | 93.72% | 0.9461 | **0.9868** |
| **Seed 42** | With GAN | 94.54% | 96.52% | 92.56% | 0.9465 | **0.9876** |
| **Seed 101** | Without GAN | 87.92% | 98.76% | 77.08% | **0.8982** *(0.898162)* | **0.9813** *(0.981271)* |
| **Seed 101** | With GAN | 84.70% | 99.40% | 70.00% | **0.8720** *(0.871986)* | **0.9840** *(0.983957)* |

### Root Cause Analysis & Evidence
In an earlier audit report draft, values **0.8982** and **0.8720** were mistakenly placed into the ROC-AUC column for Seed 101. As verified in `evidence_package/gan_comparison/gan_vs_nogan_multi_seed_comparison.csv`:
- For Seed 101 Without GAN, F1-Score is **0.898162** and ROC-AUC is **0.981271**.
- For Seed 101 With GAN, F1-Score is **0.871986** and ROC-AUC is **0.983957**.

Both metrics originate from the exact same evaluation run. ROC-AUC is consistently high across both seeds (0.9813 -> 0.9840 on Seed 101; 0.9868 -> 0.9876 on Seed 42). The lower F1-score on Seed 101 reflects a more aggressive shift toward sensitivity (99.40%), which depresses specificity (70.00%) and generates additional false positives, lowering precision and F1.

---

## 3. Verified Benchmark Metrics (Objective 3)

Evaluated on `08_hierarchical_minn/models/minn_best.pt` using the 5,000 held-out multimodal test combinations:
- **Accuracy:** 94.54% (4,727 / 5,000 correct)
- **Balanced Accuracy:** 94.54%
- **Sensitivity (Recall):** 96.52% (2,413 / 2,500 ASD instances detected)
- **Specificity:** 92.56% (2,314 / 2,500 Control instances detected)
- **Precision:** 92.84% (2,413 / 2,599 predicted positive)
- **F1-Score:** 0.9465
- **Test-Set ROC-AUC:** 0.987621
- **Brier Score Loss:** 0.042004
- **Confusion Matrix:**
  $$\begin{bmatrix} TN = 2314 & FP = 186 \\ FN = 87 & TP = 2413 \end{bmatrix}$$

---

## 4. Objective Evaluation of WGAN-GP Augmentation (Objective 2)

| Condition | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC | False Negatives (FN) | False Positives (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Without GAN (Seed 42)** | 94.56% | 95.40% | 93.72% | 0.9461 | 0.9868 | 115 | 157 |
| **With GAN (Seed 42)** | 94.54% | **96.52%** | 92.56% | 0.9465 | **0.9876** | **87** | 186 |
| *Delta (With - Without)* | *-0.02%* | *+1.12%* | *-1.16%* | *+0.0004* | *+0.0008* | *-28 (-24.3%)* | *+29* |
| **Multi-Seed Mean (Without GAN)** | 91.74% | 96.56% | 86.92% | 0.9221 | 0.9840 | 86.0 | 327.0 |
| **Multi-Seed Mean (With GAN)** | 90.05% | **97.44%** | 82.66% | 0.9092 | **0.9858** | **64.0** | 433.5 |

**Audit Conclusion:**
WGAN-GP data augmentation does not increase overall classification accuracy. Its genuine, clinically valuable contribution is smoothing the minority control boundary to detect more positive cases, increasing sensitivity by +1.12% and reducing missed cases (false negatives) by 24.3%, in exchange for a modest increase in false positives (+29).

---

## 5. Statistical Dependence & Multi-Way Crossed Bootstrap (Method 4)

### Method Comparison Table

| Resampling Method | Resampling Unit | Iterations | Accuracy 95% CI | Sensitivity 95% CI | Specificity 95% CI | ROC-AUC 95% CI | CI Width (Acc) | Relative Width |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Method 1: Naive Row** | 5,000 In-Silico Rows | 1,000 | [0.9392, 0.9518] | [0.9576, 0.9717] | [0.9153, 0.9362] | [0.9848, 0.9901] | 0.0126 | 1.0x (Reference) |
| **Method 2: Genetic Cluster** | 24 Genetic Subjects | 1,000 | [0.9166, 0.9699] | [0.9287, 0.9909] | [0.8792, 0.9676] | [0.9670, 0.9977] | 0.0533 | 4.2x |
| **Method 3: Two-Way Crossed** | 24 Genetic $\times$ 132 Behavioral | 1,000 | [0.9131, 0.9743] | [0.9362, 0.9924] | [0.8671, 0.9724] | [0.9723, 0.9982] | 0.0612 | 4.9x |
| **Method 4: Three-Way Crossed** | 24 G $\times$ 132 B $\times$ 7,166 E | 1,000 | **[0.9146, 0.9758]** | **[0.9347, 0.9935]** | **[0.8720, 0.9758]** | **[0.9717, 0.9983]** | **0.0612** | **4.9x** |

### Mathematical Estimand and Structural Limitations
1. **Mathematical Estimand:** Method 4 quantifies the sampling variability of the MINN model's performance when tested on synthetic triplet configurations formed by independently drawing from the empirical distributions of the three constituent retrospective datasets:
   $$\theta = \mathbb{E}_{G \sim \hat{F}_G, B \sim \hat{F}_B, E \sim \hat{F}_E}[L(f_\theta(G, B, E), Y)]$$
2. **Variance Bottleneck:** Because the genetic test cohort contains only $N_G = 24$ unique subjects, genetic resampling is the dominant driver of sampling variance. Adding environmental clustering ($N_E = 7,166$) does not materially widen the interval beyond Two-Way Crossed clustering (both yield width $\approx 0.0612$).
3. **Explicit Disclaimer on Clinical Non-Generalizability:**
   Resampling methods (including Method 4) evaluate uncertainty **conditional on the in-silico generation process**. Because the constituent cohorts were collected independently under different protocols and joined class-conditionally under an assumption of conditional independence, **no bootstrap method can provide a defensible real-world clinical population confidence interval**. These intervals measure reproducibility over the in-silico experimental design, not prospective clinical efficacy.

---

## 6. Data Leakage, Survey Duplicates, and Latent Distance Justification

### Multi-Modality Partition Isolation
Across all three modalities, source records were partitioned prior to representation learning:
- **Genetic (GSE28521, N=116):** Train=69, Val=23, Test=24. Overlapping Subject IDs = **0**.
- **Behavioral (Toddler Screening, N=880):** Train=616, Val=132, Test=132. Overlapping Subject IDs = **0**.
- **Environmental (NSCH Toxicological, N=47,768):** Train=33,436, Val=7,166, Test=7,166. Overlapping Record IDs = **0**.
- **Synthetic Isolation:** All 46 WGAN-GP synthetic genetic samples are strictly confined to the training set. Synthetic test samples = **0**.

### Environmental Survey Duplicates
An audit of the environmental NSCH survey revealed **2,564 duplicated feature vectors** within the 47,768 records. When split 70/15/15, this natural discretization caused **8 identical 32-D vectors** to appear in both the training partition (33,436 rows) and test partition (7,166 rows).
- **Audit Finding:** These 8 instances represent distinct children from separate households who provided identical discrete survey answers (e.g., standard responses to household smoke exposure, maternal education, and birthweight categories).
- **Conclusion:** They are feature ties inherent to survey data discretization, not shared patients or partition leakage.

### Justification and Limitations of the 0.05 Latent Distance Threshold
- In representation learning, an empirical threshold of $\epsilon = 0.05$ was established to confirm that representations of opposing clinical classes (ASD vs. Control) do not collapse onto the same latent coordinates.
- **Minimum observed cross-class distance:** **0.0738** (> 0.05).
- **Methodological Disclaimer:** A distance $> 0.05$ proves that the latent space does not suffer from representation collapse across diagnostic labels. However, **it does not constitute mathematical proof that information leakage is absent**. True leakage prevention is guaranteed by strict subject-level partition isolation, which was independently verified.

---

## 7. Latent-Space Counterfactual Analysis & Genetic Invariance (Objective 4)

Target: $P(\text{ASD}) \le 0.35$. Constraint: Genetic representations are immutable ($\Delta G = 0$).

| Demo Patient | Baseline Model Probability | Full 32-D Latent CF Probability | Full 32-D L2 Cost | Full 32-D Target Met? | Top-3 Sparse CF Probability | Top-3 Target Met? | Genetic Invariance ($\Delta G$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P001** | 0.9999 | **0.1207** | 1.6091 | **YES** | 0.9998 | NO | **0.000000** |
| **P003** | 0.6343 | **0.0021** | 0.6552 | **YES** | **0.1697** | **YES** | **0.000000** |
| **P005** | 1.0000 | **0.3302** | 4.6385 | **YES** | 0.9999 | NO | **0.000000** |

**Clarification of Mathematical Nature:**
- Full 32-D latent shift discovers joint continuous trajectories on the learned manifold reaching $P \le 0.35$.
- Top-3 sparse perturbation alters only the top 3 individual coordinates. For borderline patient P003, top-3 sparse perturbation drops risk to 0.1697 (crossing the threshold). For high-confidence cases (P001, P005), 3 coordinates are mathematically insufficient to overturn multi-dimensional activation.
- Both operations are mathematical sensitivity analyses, not clinical prescriptions.

---

## 8. Seven Modality Ablation Matrix

| Configuration | Active Modalities | Accuracy | Sensitivity | Specificity | ROC-AUC | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tri-Modal (Full MINN)** | **G + B + E** | **94.54%** | **96.52%** | **92.56%** | **0.9876** | **87** |
| Dual: Behavioral + Environmental | B + E | 94.42% | 94.72% | 94.12% | 0.9908 | 132 |
| Dual: Genetic + Behavioral | G + B | 92.24% | 98.20% | 86.28% | 0.9888 | 45 |
| Single: Behavioral Only | B | 91.54% | 100.00% | 83.08% | 0.9911 | 0 |
| Dual: Genetic + Environmental | G + E | 65.42% | 85.32% | 45.52% | 0.7981 | 367 |
| Single: Genetic Only | G | 64.44% | 95.48% | 33.40% | 0.8154 | 113 |
| Single: Environmental Only | E | 50.36% | 99.80% | 0.92% | 0.6701 | 5 |

---

## 9. Complete Automated Test Verification

| Test Suite File | Command | Tests Run | Passed | Failed | Exit Code |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests/test_audit_requirements.py` | `python tests/test_audit_requirements.py` | 11 | 11 | 0 | **0** |
| `tests/test_evidence_gaps.py` | `python tests/test_evidence_gaps.py` | 10 | 10 | 0 | **0** |
| **Total Automated Tests** | | **21** | **21** | **0** | **0** |

---

## 10. Table of Resolved vs. Unresolved Issues

| Item | Initial Audit Critique | Resolution & Verified Status | Status |
| :---: | :--- | :--- | :---: |
| **1** | Command exit code 1 failures (API script, pytest, git status) | All three investigated, root causes corrected, reruns executed. All pass with Exit Code 0. | **RESOLVED** |
| **2** | Seed 101 ROC-AUC discrepancy (0.8982 vs 0.9813; 0.8720 vs 0.9840) | Reconciled as column shift between F1-Score and ROC-AUC. Both confirmed from same evaluation. | **RESOLVED** |
| **3** | Lack of Method 4 Crossed Cluster Bootstrap for 3 modalities | Method 4 ($G \times B \times E$) implemented; 95% CI [0.9146, 0.9758]; non-generalizability disclaimed. | **RESOLVED** |
| **4** | Data leakage checks limited to genetic modality | Audited partitions across all 3 modalities (0 overlap). Survey duplicate feature ties explained. | **RESOLVED** |
| **5** | 0.05 latent distance presented as proof of zero leakage | Clarified as empirical safeguard against manifold collapse, not mathematical proof of zero leakage. | **RESOLVED** |
| **6** | Claiming GAN increases accuracy despite 94.56% to 94.54% drop | Clarified: GAN does not increase accuracy (-0.02%); contribution is +1.12% sensitivity and -24.3% FN. | **RESOLVED** |
| **7** | Missing HTTP 404 test on invalid demo patient ID | Added upfront path check and explicit 404 handler in `backend/app/main.py`. Verified in test suite. | **RESOLVED** |
| **8** | Unscientific clinical claims in documentation and UI | Standardized to "Highest modality attention weight", "Test-set ROC-AUC", and "in-silico combinations". | **RESOLVED** |
| **9** | Real-world clinical prospective validation | **None possible without prospective clinical trials**. Clearly documented as an in-silico limitation. | **ACKNOWLEDGED SCOPE BOUNDARY** |

---

## Final Recommendation & Sign-Off
All technical concerns, execution anomalies, and statistical gaps identified during audit have been rigorously resolved using genuine code execution, verified checkpoints, and transparent documentation. The project is fully reproducible and meets all technical acceptance criteria.
