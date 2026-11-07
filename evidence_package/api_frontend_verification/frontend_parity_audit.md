# Frontend / Backend Interface & Parity Verification Audit

**Audit Date:** 2026-10-30 16:39:22  
**Target:** FastAPI REST Backend + Static Client Application  

## 1. Summary of Parity Checks

| Verification Target | Requirement | Implementation Status | Evidence Location |
| :--- | :--- | :---: | :--- |
| **Attention Terminology** | Enforce "Highest modality attention weight" instead of causal "Primary driving modality" | **PASS** | `frontend/js/ui.js:347` |
| **Causal Disclaimer** | Explicitly state attention weights do not prove biological/clinical causation | **PASS** | `frontend/js/ui.js:348` |
| **Performance Metric** | Display "Test-set ROC-AUC" instead of "Clinical ROC-AUC" | **PASS** | `frontend/index.html` |
| **Latent Dimensions** | Label features as "Latent Behavioral Dim", "Latent Environmental Dim" | **PASS** | `09_explainability_benchmarking/counterfactual_generator.py` |
| **Float Precision** | Consistent percentage conversion: `(val * 100).toFixed(2) + '%'` | **PASS** | `frontend/js/ui.js:pct()` |
| **API Endpoints** | Dynamic evaluation on actual trained weights, zero hardcoded mock values | **PASS** | `backend/app/main.py` |

## 2. API Contract Verification
Every demo patient response (`/api/minn/explain/demo/P003`) returns dynamic:
1. `prediction.probability` and `prediction.classification`
2. `hierarchical_attention.level_1_intra_modality` (32 dimensions per modality)
3. `hierarchical_attention.level_2_modality_attention` (sums to 1.0000)
4. `actionable_recommendations` (minimal interventions constrained to Delta G = 0)

All results are produced dynamically by the frozen trained PyTorch checkpoint `minn_best.pt`.
