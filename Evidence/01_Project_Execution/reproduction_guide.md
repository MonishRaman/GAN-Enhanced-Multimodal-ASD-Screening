# ASD Multimodal MINN Project: Independent Execution & Reproduction Guide

**Audit Version:** 2.0 (Client Acceptance Release)  
**Execution Environment:** Windows / Linux / macOS, Python 3.10+  
**Target Architecture:** Hierarchical Attention Multimodal Integration Network (MINN)  

---

## 1. Environment Configuration

### Prerequisites
- Python 3.10.x or 3.11.x
- Git & Virtualenv

```bash
# Clone and prepare repository
git clone <repository_url>
cd GAN-3

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install exact pinned dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy fastapi uvicorn httpx pytest
```

---

## 2. Model Checkpoint Verification

Before running evaluations, verify the integrity of the primary MINN checkpoint:
- **Path:** `08_hierarchical_minn/models/minn_best.pt`
- **File Size:** `129,049 bytes`
- **SHA-256 Hash:** `bb718e18b1615c647005e6adbaa9d271cfbecfc6d149b2ecbec8b96b99d8cb40`

Verification command:
```bash
python -c "
import hashlib
from pathlib import Path
p = Path('08_hierarchical_minn/models/minn_best.pt')
h = hashlib.sha256(p.read_bytes()).hexdigest()
print('File size:', len(p.read_bytes()), 'bytes')
print('SHA-256:', h)
assert h == 'bb718e18b1615c647005e6adbaa9d271cfbecfc6d149b2ecbec8b96b99d8cb40'
print('[OK] Checkpoint verified.')
"
```

---

## 3. Step-by-Step Audit & Reproduction Commands

Run each verification script in sequence. All scripts write logs to `evidence_package/execution_logs/` and outputs to `evidence_package/`:

| Step | Script Command | Output Verification Target |
| :--- | :--- | :--- |
| **Step 1** | `python audit_scripts/1_benchmark_reproduction.py` | Reproduces **94.54% Accuracy**, **0.9876 ROC-AUC**, **TN=2314, FP=186, FN=87, TP=2413** |
| **Step 2** | `python audit_scripts/2_gan_comparison.py` | Multi-seed GAN vs no-GAN comparison; confirms **Sensitivity +1.12%**, **FN -24.3%**, Accuracy delta **-0.02%**; Seed 101 reconciliation |
| **Step 3** | `python audit_scripts/3_data_construction_and_leakage_audit.py` | 3-modality partition overlap (0 overlap across all 3), cohort reuse analysis, survey duplicate analysis, distance threshold justification |
| **Step 4** | `python audit_scripts/4_statistical_validation.py` | Compares 4 bootstrap methods including Method 4 (Three-Way Crossed Cluster: `[0.9146, 0.9758]`), with explicit non-generalizability disclaimers |
| **Step 5** | `python audit_scripts/5_attention_and_counterfactual_audit.py` | Level-1/2 dynamic attention; full 32-D (P<=0.35) vs Top-3 sparse; genetic invariance **Delta G = 0** |
| **Step 6** | `python audit_scripts/6_seven_modalities_and_api_verification.py` | 7-modality ablation (G+B+E 94.54% down to E 50.36%); FastAPI endpoints verified; HTTP 404 test on missing patient |
| **Step 7** | `python tests/test_audit_requirements.py` | 11/11 automated unit tests pass (Exit Code 0) |
| **Step 8** | `python tests/test_evidence_gaps.py` | 10/10 automated audit gap tests pass (Exit Code 0) |

---

## 4. Launching the Live Application

### Terminal 1: FastAPI Backend
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs: `http://127.0.0.1:8000/docs`
- Demo Patient Endpoint: `http://127.0.0.1:8000/api/minn/explain/demo/P001`
- Error Test (Returns HTTP 404): `http://127.0.0.1:8000/api/minn/explain/demo/INVALID_ID_999`

### Terminal 2: Frontend Client
Open `frontend/index.html` in any modern web browser or serve via:
```bash
python -m http.server 3000 --directory frontend
```
- Open `http://127.0.0.1:3000` in your web browser.
