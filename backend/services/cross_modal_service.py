"""Final feature-level cross-modal ASD inference service."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_PATH = PROJECT_ROOT / "07_final_cross_modal_asd" / "models" / "FINAL_CROSS_MODAL_ASD_MODEL.pt"

class FinalCrossModalASDNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.genetic_branch = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.behavioral_branch = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.environmental_branch = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.cross_modal_network = nn.Sequential(
            nn.Linear(288,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU()
        )
        self.classifier = nn.Linear(32,1)

    def forward(self, genetic, behavioral, environmental):
        g = self.genetic_branch(genetic)
        b = self.behavioral_branch(behavioral)
        e = self.environmental_branch(environmental)
        x = torch.cat([g,b,e,torch.abs(g-b),torch.abs(g-e),torch.abs(b-e),g*b,g*e,b*e], dim=1)
        joint = self.cross_modal_network(x)
        return self.classifier(joint), joint

class CrossModalService:
    def __init__(self) -> None:
        self.model_path = FINAL_MODEL_PATH
        self.threshold = 0.50
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Final Cross-Modal ASD model not found:\n{self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Final Cross-Modal checkpoint does not contain 'model_state_dict'.")
        self.model = FinalCrossModalASDNetwork().to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    @staticmethod
    def _validate(value: Any, modality: str) -> np.ndarray:
        a = np.asarray(value, dtype=np.float32)
        if a.ndim == 2:
            if a.shape[0] != 1:
                raise ValueError(f"{modality} must contain exactly one patient row; received shape {a.shape}.")
            a = a[0]
        if a.ndim != 1 or a.shape[0] != 32:
            raise ValueError(f"{modality} representation must be exactly 32-D; received shape {a.shape}.")
        if not np.isfinite(a).all():
            raise ValueError(f"{modality} representation contains NaN or infinite values.")
        return a

    @torch.no_grad()
    def predict(self, genetic_32d: Any, behavioral_32d: Any, environmental_32d: Any) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Final Cross-Modal ASD model is not loaded.")
        g = self._validate(genetic_32d, "Genetic")
        b = self._validate(behavioral_32d, "Behavioral")
        e = self._validate(environmental_32d, "Environmental")
        gt = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
        bt = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
        et = torch.tensor(e, dtype=torch.float32).unsqueeze(0)
        logits, joint = self.model(gt, bt, et)
        probability = float(torch.sigmoid(logits).item())
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid probability returned by final model: {probability}")
        numeric = int(probability >= self.threshold)
        return {
            "probability": probability,
            "prediction": "ASD" if numeric else "NON-ASD",
            "prediction_numeric": numeric,
            "threshold": self.threshold,
            "model": "Final Cross-Modal ASD Network",
            "joint_embedding_dimension": int(joint.shape[1]),
            "modalities_used": ["genetic", "behavioral", "environmental"],
        }

    def info(self) -> Dict[str, Any]:
        return {
            "model": "Final Cross-Modal ASD Network",
            "task": "ASD vs NON-ASD",
            "genetic_input_dimension": 32,
            "behavioral_input_dimension": 32,
            "environmental_input_dimension": 32,
            "joint_embedding_dimension": 32,
            "threshold": self.threshold,
            "device": str(DEVICE),
            "model_path": str(self.model_path),
        }

cross_modal_service = CrossModalService()
__all__ = ["CrossModalService", "FinalCrossModalASDNetwork", "cross_modal_service"]
