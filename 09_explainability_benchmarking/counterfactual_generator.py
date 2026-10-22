"""
Objective 4: Counterfactual Explanation Generator for Multimodal ASD Screening.

Computes the minimal actionable perturbation (Delta_x) to behavioral and environmental
features required to transition a patient's prediction from ASD to Non-ASD.
"""

from pathlib import Path
import sys
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Resolve paths dynamically relative to this file
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT / "08_hierarchical_minn") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "08_hierarchical_minn"))

from models.hierarchical_minn import HierarchicalAttentionMINN

MODEL_PATH = PROJECT_ROOT / "08_hierarchical_minn" / "models" / "minn_best.pt"

class CounterfactualGenerator:
    """
    Gradient-based counterfactual explanation generator for the trained MINN.
    Finds minimal, realistic perturbations in actionable modalities (Behavioral & Environmental)
    to shift predicted ASD probability below the clinical decision boundary.
    """
    def __init__(self, model_path: Path = MODEL_PATH, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = HierarchicalAttentionMINN(in_features=32, branch_hidden=64, branch_out=32).to(self.device)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained MINN model checkpoint not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Freeze model parameters during counterfactual search
        for p in self.model.parameters():
            p.requires_grad = False

    def generate_counterfactual(
        self,
        genetic_32d: np.ndarray,
        behavioral_32d: np.ndarray,
        environmental_32d: np.ndarray,
        patient_id: str = "P_SAMPLE",
        target_prob: float = 0.35,
        max_iter: int = 150,
        lr: float = 0.05,
        lambda_pred: float = 20.0,
        lambda_l1: float = 0.01
    ) -> Dict[str, Any]:
        """
        Solves:
            min_{Delta_b, Delta_e} ||Delta||_2^2 + lambda_l1 ||Delta||_1 + lambda_pred * max(0, logit - logit_target)^2
        Uses logit-space optimization to prevent gradient vanishing on saturated predictions.
        """
        g_t = torch.tensor(genetic_32d, dtype=torch.float32).reshape(1, -1).to(self.device)
        b_t = torch.tensor(behavioral_32d, dtype=torch.float32).reshape(1, -1).to(self.device)
        e_t = torch.tensor(environmental_32d, dtype=torch.float32).reshape(1, -1).to(self.device)

        with torch.no_grad():
            orig_logit_t, orig_prob_t = self.model(g_t, b_t, e_t)
        orig_prob = float(orig_prob_t.cpu().item())
        orig_logit = float(orig_logit_t.cpu().item())

        if orig_prob < 0.50:
            return {
                "patient_id": patient_id,
                "status": "ALREADY_NEGATIVE",
                "original_probability": round(orig_prob, 4),
                "message": f"Patient {patient_id} already has a non-ASD prediction ({orig_prob:.4f} < 0.50). No counterfactual required."
            }

        # Safe logit target to prevent numerical overflow
        safe_target_prob = min(max(target_prob, 1e-4), 0.49)
        target_logit = float(np.log(safe_target_prob / (1.0 - safe_target_prob)))

        # Initialize perturbations for actionable modalities
        delta_b = nn.Parameter(torch.zeros_like(b_t, requires_grad=True))
        delta_e = nn.Parameter(torch.zeros_like(e_t, requires_grad=True))

        optimizer = Adam([delta_b, delta_e], lr=lr)

        best_loss = float("inf")
        best_cf_prob = orig_prob
        best_delta_b = delta_b.detach().clone()
        best_delta_e = delta_e.detach().clone()

        for step in range(max_iter):
            optimizer.zero_grad()
            b_cf = b_t + delta_b
            e_cf = e_t + delta_e

            cf_logits, cf_prob = self.model(g_t, b_cf, e_cf)

            # Prediction penalty in logit space (prevents vanishing gradient at extreme prob values)
            prob_penalty = torch.relu(cf_logits - target_logit) ** 2
            # Distance penalties (L2 and L1)
            dist_l2 = torch.norm(delta_b, p=2) ** 2 + torch.norm(delta_e, p=2) ** 2
            dist_l1 = torch.norm(delta_b, p=1) + torch.norm(delta_e, p=1)

            total_loss = (lambda_pred * prob_penalty) + dist_l2 + (lambda_l1 * dist_l1)
            total_loss.backward()
            optimizer.step()

            current_prob = float(cf_prob.item())
            if current_prob <= best_cf_prob:
                best_cf_prob = current_prob
                best_delta_b = delta_b.detach().clone()
                best_delta_e = delta_e.detach().clone()

            # Early stopping if target probability is fully satisfied
            if current_prob <= target_prob:
                break

        if best_delta_b is None:
            best_delta_b = delta_b.detach().clone()
        if best_delta_e is None:
            best_delta_e = delta_e.detach().clone()

        # Compute perturbation metrics
        delta_b_np = best_delta_b.cpu().numpy().flatten()
        delta_e_np = best_delta_e.cpu().numpy().flatten()

        l2_dist = float(np.sqrt(np.sum(delta_b_np ** 2) + np.sum(delta_e_np ** 2)))
        l1_dist = float(np.sum(np.abs(delta_b_np)) + np.sum(np.abs(delta_e_np)))

        # Find top altered dimensions
        top_b_perturb = np.argsort(np.abs(delta_b_np))[::-1][:3]
        top_e_perturb = np.argsort(np.abs(delta_e_np))[::-1][:3]

        if best_cf_prob <= target_prob:
            status = "COUNTERFACTUAL_FOUND"
        elif best_cf_prob < 0.50:
            status = "DIAGNOSTIC_FLIP_ACHIEVED"
        else:
            status = "PARTIAL_REDUCTION_ONLY"

        classification = "Control (Non-ASD)" if best_cf_prob < 0.50 else "ASD"

        if status in ("COUNTERFACTUAL_FOUND", "DIAGNOSTIC_FLIP_ACHIEVED"):
            clinical_summary = (
                f"For patient {patient_id} (original risk {orig_prob*100:.1f}%), a target probability of {best_cf_prob*100:.1f}% "
                f"is achievable through an L2 intervention distance of {l2_dist:.2f}, primarily requiring adjustments "
                f"to Behavioral Dimensions {[int(x+1) for x in top_b_perturb]} and Environmental Exposures {[int(x+1) for x in top_e_perturb]}."
            )
        else:
            clinical_summary = (
                f"For patient {patient_id} (original risk {orig_prob*100:.1f}%), optimization achieved a partial reduction to {best_cf_prob*100:.1f}%. "
                f"Due to high non-actionable genetic risk, supplementary clinical consultation is recommended."
            )

        return {
            "patient_id": patient_id,
            "status": status,
            "original_diagnosis": {
                "probability": round(orig_prob, 4),
                "classification": "ASD"
            },
            "counterfactual_outcome": {
                "probability": round(best_cf_prob, 4),
                "classification": classification,
                "probability_reduction": round(orig_prob - best_cf_prob, 4),
                "target_threshold": target_prob
            },
            "perturbation_cost": {
                "l2_distance": round(l2_dist, 4),
                "l1_distance": round(l1_dist, 4)
            },
            "actionable_recommendations": {
                "behavioral_adjustments": [
                    {"dimension": int(idx + 1), "recommended_shift": round(float(delta_b_np[idx]), 4)}
                    for idx in top_b_perturb
                ],
                "environmental_adjustments": [
                    {"dimension": int(idx + 1), "recommended_shift": round(float(delta_e_np[idx]), 4)}
                    for idx in top_e_perturb
                ]
            },
            "clinical_summary": clinical_summary
        }

