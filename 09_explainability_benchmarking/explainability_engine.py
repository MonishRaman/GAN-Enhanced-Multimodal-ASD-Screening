"""
Objective 4: Hierarchical Attention Explainability Engine.

Extracts feature-level and modality-level attention explanations
from the trained Hierarchical Attention MINN for clinical interpretability.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

import sys
PROJECT_ROOT = Path("d:/GAN-Enhanced-Multimodal-ASD-Screening")
if str(PROJECT_ROOT / "08_hierarchical_minn") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "08_hierarchical_minn"))

from models.hierarchical_minn import HierarchicalAttentionMINN
MODEL_PATH = PROJECT_ROOT / "08_hierarchical_minn" / "models" / "minn_best.pt"

class HierarchicalExplainabilityEngine:
    """
    Inference & Explanation engine for the Hierarchical Attention MINN.
    Provides diagnostic probability, confidence, dynamic modality weights,
    and intra-modality feature saliency.
    """
    def __init__(self, model_path: Path = MODEL_PATH, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = HierarchicalAttentionMINN(in_features=32, branch_hidden=64, branch_out=32).to(self.device)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained MINN model not found at {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def explain_patient(
        self,
        genetic_32d: np.ndarray,
        behavioral_32d: np.ndarray,
        environmental_32d: np.ndarray,
        patient_id: str = "P_SAMPLE",
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Generates full explainability output for a single patient profile.
        """
        # Ensure 2D tensor (1, 32)
        g_t = torch.tensor(genetic_32d, dtype=torch.float32).reshape(1, -1).to(self.device)
        b_t = torch.tensor(behavioral_32d, dtype=torch.float32).reshape(1, -1).to(self.device)
        e_t = torch.tensor(environmental_32d, dtype=torch.float32).reshape(1, -1).to(self.device)

        with torch.no_grad():
            logits, probs, att_dict = self.model(g_t, b_t, e_t, return_attention=True)

        prob = float(probs.cpu().item())
        pred_class = "ASD" if prob >= threshold else "Control"
        confidence = prob if pred_class == "ASD" else (1.0 - prob)

        # Dynamic Inter-Modality Attention (Level 2)
        beta = att_dict["inter_modality"].cpu().numpy().flatten()
        beta_genetic = float(beta[0])
        beta_behavioral = float(beta[1])
        beta_environmental = float(beta[2])

        # Level 1 Feature Attention (Top-5 salient dimensions per modality)
        alpha_g = att_dict["intra_genetic"].cpu().numpy().flatten()
        alpha_b = att_dict["intra_behavioral"].cpu().numpy().flatten()
        alpha_e = att_dict["intra_environmental"].cpu().numpy().flatten()

        top_g_dims = np.argsort(alpha_g)[::-1][:5].tolist()
        top_b_dims = np.argsort(alpha_b)[::-1][:5].tolist()
        top_e_dims = np.argsort(alpha_e)[::-1][:5].tolist()

        return {
            "patient_id": patient_id,
            "prediction": {
                "asd_probability": round(prob, 4),
                "classification": pred_class,
                "confidence": round(confidence, 4),
                "decision_threshold": threshold
            },
            "modality_attention_level_2": {
                "genetic_weight": round(beta_genetic, 4),
                "behavioral_weight": round(beta_behavioral, 4),
                "environmental_weight": round(beta_environmental, 4),
                "primary_driving_modality": ["Genetic", "Behavioral", "Environmental"][int(np.argmax(beta))]
            },
            "feature_saliency_level_1": {
                "top_genetic_dimensions": [{"dimension": int(idx + 1), "attention_weight": round(float(alpha_g[idx]), 4)} for idx in top_g_dims],
                "top_behavioral_dimensions": [{"dimension": int(idx + 1), "attention_weight": round(float(alpha_b[idx]), 4)} for idx in top_b_dims],
                "top_environmental_dimensions": [{"dimension": int(idx + 1), "attention_weight": round(float(alpha_e[idx]), 4)} for idx in top_e_dims]
            },
            "clinical_interpretation": (
                f"Patient {patient_id} is screened as {pred_class} with {confidence*100:.1f}% confidence. "
                f"Diagnostic decision was primarily driven by the {['Genetic', 'Behavioral', 'Environmental'][int(np.argmax(beta))]} modality "
                f"({beta[int(np.argmax(beta))]*100:.1f}% dynamic weight)."
            )
        }
