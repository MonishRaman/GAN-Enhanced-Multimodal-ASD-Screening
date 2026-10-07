"""
MINN Service: Objective 3 & 4 Hierarchical Attention & Counterfactual Inference.

Provides end-to-end inference using the trained Hierarchical Attention MINN,
returning dynamic patient-specific modality attention weights, feature saliency,
and actionable counterfactual recommendations.
"""

from pathlib import Path
import sys
from typing import Dict, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "08_hierarchical_minn") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "08_hierarchical_minn"))
if str(PROJECT_ROOT / "09_explainability_benchmarking") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "09_explainability_benchmarking"))

from explainability_engine import HierarchicalExplainabilityEngine
from counterfactual_generator import CounterfactualGenerator

class MINNService:
    def __init__(self):
        self.explainer = HierarchicalExplainabilityEngine()
        self.cf_generator = CounterfactualGenerator()

    def predict_patient(
        self,
        genetic_32d: np.ndarray,
        behavioral_32d: np.ndarray,
        environmental_32d: np.ndarray,
        patient_id: str = "PATIENT_DEMO",
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Runs MINN inference and extracts hierarchical attention weights.
        """
        return self.explainer.explain_patient(
            genetic_32d=genetic_32d,
            behavioral_32d=behavioral_32d,
            environmental_32d=environmental_32d,
            patient_id=patient_id,
            threshold=threshold
        )

    def counterfactual_analysis(
        self,
        genetic_32d: np.ndarray,
        behavioral_32d: np.ndarray,
        environmental_32d: np.ndarray,
        patient_id: str = "PATIENT_DEMO",
        target_prob: float = 0.35
    ) -> Dict[str, Any]:
        """
        Computes minimal actionable interventions to flip ASD prediction.
        """
        return self.cf_generator.generate_counterfactual(
            genetic_32d=genetic_32d,
            behavioral_32d=behavioral_32d,
            environmental_32d=environmental_32d,
            patient_id=patient_id,
            target_prob=target_prob
        )

minn_service = MINNService()
