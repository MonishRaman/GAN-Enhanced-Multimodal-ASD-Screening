"""
audit_scripts/check_parity.py

Compares outputs across:
1. Direct checkpoint inference
2. Backend /api/minn/predict endpoint
3. Backend /api/minn/explain/demo/P001 endpoint
4. Frontend display formatting
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "08_hierarchical_minn"))
sys.path.append(str(PROJECT_ROOT / "backend"))

from backend.app.main import app
from models.hierarchical_minn import HierarchicalAttentionMINN
from backend.services.genetic_service import GeneticService
from backend.services.behavioral_service import BehavioralService
from backend.services.environmental_service import EnvironmentalService

def pct(v):
    if v is None: return "N/A"
    if v <= 1: v = v * 100
    return f"{v:.2f}%"

def main():
    client = TestClient(app)

    p_dir = PROJECT_ROOT / "05_multimodal_fusion" / "M4_demo_patients" / "P001"
    g_df = pd.read_csv(p_dir / "genetic_input.csv")
    b_df = pd.read_csv(p_dir / "behavioral_input.csv")
    e_df = pd.read_csv(p_dir / "environmental_input.csv")

    # 1. Direct Model Inference
    gs = GeneticService()
    bs = BehavioralService()
    es = EnvironmentalService()

    g_res = gs.process(g_df)
    b_res = bs.predict(b_df)
    e_res = es.process(p_dir / "environmental_input.csv")

    g_32d = np.array(g_res["aligned_embedding_32"], dtype=np.float32)
    b_32d = np.array(b_res["aligned_embedding_32"], dtype=np.float32)
    e_32d = np.array(e_res["aligned_embedding_32"], dtype=np.float32)

    model = HierarchicalAttentionMINN(in_features=32, branch_hidden=64, branch_out=32)
    model.load_state_dict(torch.load(PROJECT_ROOT / "08_hierarchical_minn/models/minn_best.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        g_t = torch.tensor(g_32d, dtype=torch.float32)
        b_t = torch.tensor(b_32d, dtype=torch.float32)
        e_t = torch.tensor(e_32d, dtype=torch.float32)
        _, direct_prob_t = model(g_t, b_t, e_t)
        direct_prob = float(direct_prob_t.item())

    # 2. Backend Prediction endpoint
    files = {
        "genetic_file": ("genetic.csv", (p_dir / "genetic_input.csv").read_bytes(), "text/csv"),
        "behavioral_file": ("behavioral.csv", (p_dir / "behavioral_input.csv").read_bytes(), "text/csv"),
        "environmental_file": ("environmental.csv", (p_dir / "environmental_input.csv").read_bytes(), "text/csv")
    }
    res_pred = client.post("/api/minn/predict", files=files)
    pred_json = res_pred.json()

    # 3. Backend Explain endpoint
    res_exp = client.get("/api/minn/explain/demo/P001")
    exp_json = res_exp.json()

    print("=" * 70)
    print("APPLICATION PARITY COMPARISON FOR DEMO PATIENT P001")
    print("=" * 70)
    print(f"1. Direct Model Inference:      Probability = {direct_prob:.6f}")
    print(f"2. Backend /predict Endpoint:    Probability = {pred_json['asd_probability']:.6f}, Confidence = {pred_json['confidence']:.6f}, Class = {pred_json['predicted_class']}")
    print(f"3. Backend /explain/demo/P001:  Probability = {exp_json['prediction']['probability']:.6f}, Confidence = {exp_json['prediction']['confidence']:.6f}, Class = {exp_json['prediction']['class']}")
    print(f"4. Frontend UI Display Format:   Probability = {pct(pred_json['asd_probability'])}, Confidence = {pct(pred_json['confidence'])}")
    print("\n--- ATTENTION WEIGHTS (BETA) ---")
    print(f"Backend /predict: {pred_json['modality_attention']}")
    print(f"Backend /explain: {exp_json['explanation']['modality_attention']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
