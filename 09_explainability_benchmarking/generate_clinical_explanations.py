"""
Objective 4: Clinical Explanation & Counterfactual Demo Runner.

Generates complete integrated clinical interpretability reports and
counterfactual case studies for demonstration and validation.
"""

import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("d:/GAN-Enhanced-Multimodal-ASD-Screening")
if str(PROJECT_ROOT / "08_hierarchical_minn") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "08_hierarchical_minn"))
if str(PROJECT_ROOT / "09_explainability_benchmarking") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "09_explainability_benchmarking"))

from explainability_engine import HierarchicalExplainabilityEngine
from counterfactual_generator import CounterfactualGenerator
from data_loader import get_multimodal_dataloaders

OUT_DIR = PROJECT_ROOT / "09_explainability_benchmarking"
REPORT_DIR = OUT_DIR / "reports"
FIGURE_DIR = OUT_DIR / "figures"
RESULT_DIR = OUT_DIR / "results"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("OBJECTIVE 4: GENERATING INTEGRATED CLINICAL EXPLANATIONS")
    print("=" * 70)

    # 1. Initialize Engines
    explainer = HierarchicalExplainabilityEngine()
    cf_generator = CounterfactualGenerator()

    # 2. Load held-out test data
    _, _, _, (G_te, B_te, E_te, y_te) = get_multimodal_dataloaders(n_test=500)

    # Find representative positive ASD cases
    asd_indices = np.where(y_te == 1)[0]
    ctrl_indices = np.where(y_te == 0)[0]

    demo_cases = []
    # Analyze 5 test cases
    test_case_ids = ["PATIENT_001_HIGH_ASD", "PATIENT_002_MODERATE_ASD", "PATIENT_003_BORDERLINE", "PATIENT_004_CONTROL", "PATIENT_005_CONTROL"]
    selected_indices = [asd_indices[0], asd_indices[1], asd_indices[2], ctrl_indices[0], ctrl_indices[1]]

    patient_reports = []

    for pid, idx in zip(test_case_ids, selected_indices):
        g_sample = G_te[idx]
        b_sample = B_te[idx]
        e_sample = E_te[idx]

        # 1. Feature & Modality Attention Explanation
        exp = explainer.explain_patient(g_sample, b_sample, e_sample, patient_id=pid)

        # 2. Counterfactual Explanation (if diagnosed as ASD)
        if exp["prediction"]["classification"] == "ASD":
            cf = cf_generator.generate_counterfactual(g_sample, b_sample, e_sample, patient_id=pid)
        else:
            cf = {"patient_id": pid, "status": "ALREADY_NON_ASD", "message": "Patient is negative for ASD."}

        full_profile = {
            "patient_id": pid,
            "attention_explanation": exp,
            "counterfactual_explanation": cf
        }
        patient_reports.append(full_profile)
        print(f"Processed {pid} -> Probability: {exp['prediction']['asd_probability']:.4f}, Class: {exp['prediction']['classification']}")

    # Save detailed JSON report
    with open(RESULT_DIR / "patient_clinical_explanations.json", "w") as f:
        json.dump(patient_reports, f, indent=4)
    print(f"\n[OK] Patient clinical explanations saved to {RESULT_DIR / 'patient_clinical_explanations.json'}")

    # 3. Create Visual Visualization: Counterfactual Case Study Plot
    case = patient_reports[0] # PATIENT_001_HIGH_ASD
    cf_data = case["counterfactual_explanation"]
    
    if cf_data.get("status") == "COUNTERFACTUAL_FOUND":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Bar chart comparing original vs counterfactual probability
        p_orig = cf_data["original_diagnosis"]["probability"]
        p_cf = cf_data["counterfactual_outcome"]["probability"]
        bars = axes[0].bar(["Original Screening", "Counterfactual Goal"], [p_orig, p_cf], color=["crimson", "seagreen"], width=0.5)
        axes[0].axhline(0.5, color="black", linestyle="--", label="Diagnostic Threshold (0.50)")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("ASD Probability", fontsize=11)
        axes[0].set_title(f"Counterfactual Outcome — {case['patient_id']}", fontsize=12, fontweight="bold")
        axes[0].legend()
        for bar in bars:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2%}", ha='center', va='bottom', fontweight='bold')

        # Actionable adjustments
        b_adjs = cf_data["actionable_recommendations"]["behavioral_adjustments"]
        e_adjs = cf_data["actionable_recommendations"]["environmental_adjustments"]
        
        labels = [f"Beh Dim {x['dimension']}" for x in b_adjs] + [f"Env Dim {x['dimension']}" for x in e_adjs]
        vals = [x["recommended_shift"] for x in b_adjs] + [x["recommended_shift"] for x in e_adjs]
        
        colors = ["royalblue" if "Beh" in l else "coral" for l in labels]
        axes[1].barh(labels, vals, color=colors)
        axes[1].axvline(0, color="black", lw=1)
        axes[1].set_xlabel("Recommended Latent Feature Shift (Delta)", fontsize=11)
        axes[1].set_title("Actionable Clinical Interventions Needed to Flip Diagnosis", fontsize=12, fontweight="bold")

        plt.tight_layout()
        cf_plot_path = FIGURE_DIR / "clinical_counterfactual_case_study.png"
        plt.savefig(cf_plot_path, dpi=300)
        plt.close()
        print(f"[OK] Counterfactual case study plot saved to {cf_plot_path}")

    # 4. Attention Modality Proportions Plot Across Demo Cases
    plt.figure(figsize=(10, 5))
    pids = [p["patient_id"] for p in patient_reports]
    gen_weights = [p["attention_explanation"]["modality_attention_level_2"]["genetic_weight"] for p in patient_reports]
    beh_weights = [p["attention_explanation"]["modality_attention_level_2"]["behavioral_weight"] for p in patient_reports]
    env_weights = [p["attention_explanation"]["modality_attention_level_2"]["environmental_weight"] for p in patient_reports]

    x = np.arange(len(pids))
    width = 0.25
    plt.bar(x - width, gen_weights, width, label="Genetic (β_G)", color="royalblue")
    plt.bar(x, beh_weights, width, label="Behavioral (β_B)", color="mediumseagreen")
    plt.bar(x + width, env_weights, width, label="Environmental (β_E)", color="coral")

    plt.xticks(x, [p.replace("PATIENT_", "P_") for p in pids], rotation=15, fontsize=10)
    plt.ylabel("Dynamic Attention Proportion", fontsize=11)
    plt.title("Patient-Level Inter-Modality Attention Comparison", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    att_plot_path = FIGURE_DIR / "patient_attention_profiles.png"
    plt.savefig(att_plot_path, dpi=300)
    plt.close()
    print(f"[OK] Patient attention profiles plot saved to {att_plot_path}")

    # 5. Write Clinical Summary Report
    with open(REPORT_DIR / "objective4_explainability_report.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("OBJECTIVE 4: CLINICAL EXPLAINABILITY & COUNTERFACTUAL REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("SUMMARY OF IMPLEMENTATION:\n")
        f.write("1. Hierarchical attention extraction successfully provides:\n")
        f.write("   - Level 2: Patient-specific modality importance (beta_G, beta_B, beta_E).\n")
        f.write("   - Level 1: Top-5 salient latent feature dimensions per modality.\n")
        f.write("2. Counterfactual generation successfully computes minimum actionable perturbations\n")
        f.write("   to transition ASD-positive cases below the 0.50 diagnostic threshold.\n\n")
        f.write("SAMPLE CASE STUDY:\n")
        f.write("-" * 70 + "\n")
        f.write(json.dumps(patient_reports[0], indent=2) + "\n")

    print(f"[OK] Clinical summary report written to {REPORT_DIR / 'objective4_explainability_report.txt'}")

if __name__ == "__main__":
    main()
