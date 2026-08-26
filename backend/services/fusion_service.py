"""
M3.2 Multimodal Knowledge-Level Evidence Fusion Service.

Frozen inference design:
    Genetic       -> expert probability
    Behavioral    -> expert probability
    Environmental -> expert probability

Final probability:
    0.3194 * genetic_probability
  + 0.4002 * behavioral_probability
  + 0.2804 * environmental_probability

Decision threshold:
    0.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUSION_DIR = PROJECT_ROOT / "05_multimodal_fusion"


GENETIC_EXPERT_PATH = FUSION_DIR / "M3_1_genetic_expert.pt"
BEHAVIORAL_EXPERT_PATH = FUSION_DIR / "M3_1_behavioral_expert.pt"
ENVIRONMENTAL_EXPERT_PATH = FUSION_DIR / "M3_1_environmental_expert.pt"

GENETIC_SCALER_PATH = FUSION_DIR / "M3_1_genetic_scaler.joblib"
BEHAVIORAL_SCALER_PATH = FUSION_DIR / "M3_1_behavioral_scaler.joblib"
ENVIRONMENTAL_SCALER_PATH = FUSION_DIR / "M3_1_environmental_scaler.joblib"


# ---------------------------------------------------------------------------
# FROZEN M3.2 CONFIGURATION
# ---------------------------------------------------------------------------

GENETIC_WEIGHT = 0.3194
BEHAVIORAL_WEIGHT = 0.4002
ENVIRONMENTAL_WEIGHT = 0.2804

DECISION_THRESHOLD = 0.5

MODALITY_DIMENSION = 32


# ---------------------------------------------------------------------------
# EXPERT ARCHITECTURE
# ---------------------------------------------------------------------------

class MultimodalExpert(nn.Module):
    """
    Architecture used by the M3_1 modality expert checkpoints.

    32
      ↓
    Linear 32 → 64
      ↓
    BatchNorm
      ↓
    ReLU
      ↓
    Dropout
      ↓
    Linear 64 → 32
      ↓
    ReLU
      ↓
    Linear 32 → 1
    """

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(32, 64),       # network.0
            nn.BatchNorm1d(64),      # network.1
            nn.ReLU(),               # network.2
            nn.Dropout(0.2),         # network.3

            nn.Linear(64, 32),       # network.4
            nn.Identity(),           # network.5
            nn.Identity(),           # network.6
            nn.Linear(32, 1),        # network.7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# FUSION SERVICE
# ---------------------------------------------------------------------------

class FusionService:
    """
    Frozen M3.2 weighted-probability multimodal fusion.

    Inputs:
        genetic       : 32-dimensional aligned representation
        behavioral    : 32-dimensional aligned representation
        environmental: 32-dimensional aligned representation

    Outputs:
        modality probabilities
        weighted fused probability
        final ASD / NON-ASD decision
    """

    def __init__(self) -> None:

        self.device = torch.device("cpu")

        # ---------------------------------------------------------------
        # Load expert models
        # ---------------------------------------------------------------

        self.genetic_expert = self._load_expert(
            GENETIC_EXPERT_PATH,
            "genetic",
        )

        self.behavioral_expert = self._load_expert(
            BEHAVIORAL_EXPERT_PATH,
            "behavioral",
        )

        self.environmental_expert = self._load_expert(
            ENVIRONMENTAL_EXPERT_PATH,
            "environmental",
        )

        # ---------------------------------------------------------------
        # Load modality scalers
        # ---------------------------------------------------------------

        self.genetic_scaler = self._load_scaler(
            GENETIC_SCALER_PATH,
            "genetic",
        )

        self.behavioral_scaler = self._load_scaler(
            BEHAVIORAL_SCALER_PATH,
            "behavioral",
        )

        self.environmental_scaler = self._load_scaler(
            ENVIRONMENTAL_SCALER_PATH,
            "environmental",
        )

    # ------------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------------

    def _load_expert(
        self,
        path: Path,
        name: str,
    ) -> nn.Module:

        if not path.exists():
            raise FileNotFoundError(
                f"{name.capitalize()} expert checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "state_dict",
                checkpoint.get(
                    "model_state_dict",
                    checkpoint,
                ),
            )
        else:
            state_dict = checkpoint

        model = MultimodalExpert().to(self.device)

        # Handle DataParallel checkpoints if present.
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]

            cleaned_state_dict[key] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        model.eval()

        return model

    def _load_scaler(
        self,
        path: Path,
        name: str,
    ) -> Any:

        if not path.exists():
            raise FileNotFoundError(
                f"{name.capitalize()} scaler not found: {path}"
            )

        return joblib.load(path)

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_embedding(
        embedding: Any,
        name: str,
    ) -> np.ndarray:

        if isinstance(embedding, torch.Tensor):
            array = (
                embedding
                .detach()
                .cpu()
                .numpy()
            )
        else:
            array = np.asarray(
                embedding,
                dtype=np.float32,
            )

        array = np.asarray(
            array,
            dtype=np.float32,
        )

        # Accept:
        #   (32,)
        #   (1, 32)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError(
                f"{name} embedding must be 1-D or 2-D; "
                f"received shape {array.shape}"
            )

        if array.shape[1] != MODALITY_DIMENSION:
            raise ValueError(
                f"{name} embedding must have "
                f"{MODALITY_DIMENSION} features; "
                f"received {array.shape[1]}"
            )

        return array

    # ------------------------------------------------------------------
    # EXPERT PREDICTION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _expert_probability(
        self,
        embedding: Any,
        scaler: Any,
        model: nn.Module,
        name: str,
    ) -> float:

        array = self._prepare_embedding(
            embedding,
            name,
        )

        scaled = scaler.transform(array)

        tensor = torch.tensor(
            scaled,
            dtype=torch.float32,
            device=self.device,
        )

        logits = model(tensor)

        probability = torch.sigmoid(
            logits
        ).detach().cpu().numpy().reshape(-1)

        if probability.size != 1:
            raise ValueError(
                f"{name} expert returned "
                f"{probability.size} probabilities; "
                f"expected exactly one."
            )

        return float(probability[0])

    # ------------------------------------------------------------------
    # PUBLIC FUSION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        genetic: Any,
        behavioral: Any,
        environmental: Any,
    ) -> dict[str, Any]:

        genetic_probability = self._expert_probability(
            genetic,
            self.genetic_scaler,
            self.genetic_expert,
            "genetic",
        )

        behavioral_probability = self._expert_probability(
            behavioral,
            self.behavioral_scaler,
            self.behavioral_expert,
            "behavioral",
        )

        environmental_probability = self._expert_probability(
            environmental,
            self.environmental_scaler,
            self.environmental_expert,
            "environmental",
        )

        # ---------------------------------------------------------------
        # M3.2 WEIGHTED PROBABILITY EVIDENCE FUSION
        # ---------------------------------------------------------------

        fused_probability = (
            GENETIC_WEIGHT * genetic_probability
            + BEHAVIORAL_WEIGHT * behavioral_probability
            + ENVIRONMENTAL_WEIGHT * environmental_probability
        )

        prediction_numeric = int(
            fused_probability >= DECISION_THRESHOLD
        )

        prediction = (
            "ASD"
            if prediction_numeric == 1
            else "NON-ASD"
        )

        return {
            "probability": float(fused_probability),
            "prediction": prediction,
            "prediction_numeric": prediction_numeric,
            "modalities": {
                "genetic": {
                    "probability": genetic_probability,
                    "weight": GENETIC_WEIGHT,
                },
                "behavioral": {
                    "probability": behavioral_probability,
                    "weight": BEHAVIORAL_WEIGHT,
                },
                "environmental": {
                    "probability": environmental_probability,
                    "weight": ENVIRONMENTAL_WEIGHT,
                },
            },
            "fusion_method": "weighted probability evidence fusion",
            "decision_threshold": DECISION_THRESHOLD,
        }