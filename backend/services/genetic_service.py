"""
Genetic inference service.

Frozen pipeline:

    genetic input
        ↓
    exact 1000 genetic features
        ↓
    frozen Genetic Transformer
        ↓
    128-D genetic representation
        ↓
    frozen 128 → 32 alignment
        ↓
    32-D genetic representation

No training is performed.
No checkpoint is modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SERVICE_FILE = Path(__file__).resolve()

BACKEND_DIR = SERVICE_FILE.parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

GENETIC_MODEL_DIR = PROJECT_ROOT / "Genetic_Model"

TRANSFORMER_PATH = (
    GENETIC_MODEL_DIR
    / "03_representation_learning"
    / "genetic_transformer_128d.pt"
)

ALIGNMENT_PATH = (
    GENETIC_MODEL_DIR
    / "04_cross_modal_alignment"
    / "genetic_128_to_32_alignment.pt"
)


# ---------------------------------------------------------------------
# Genetic Transformer
# ---------------------------------------------------------------------

class GeneticTransformer(nn.Module):
    """
    Exact architecture represented by
    genetic_transformer_128d.pt.

    Input:
        one scalar genetic feature

    Feature embedding:
        1 -> 128

    Transformer:
        2 encoder layers
        d_model = 128
        nhead = 8

    Output projection:
        128 -> 128
    """

    def __init__(self) -> None:
        super().__init__()

        self.feature_embedding = nn.Linear(1, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        # [batch, 1000]

        x = x.unsqueeze(-1)

        # [batch, 1000, 1]
        x = self.feature_embedding(x)

        # [batch, 1000, 128]
        x = self.transformer(x)

        # Aggregate genetic-feature representations.
        x = x.mean(dim=1)

        # [batch, 128]
        x = self.output_projection(x)

        return x


# ---------------------------------------------------------------------
# Genetic Alignment
# ---------------------------------------------------------------------

class GeneticAlignment(nn.Module):
    """
    Exact 128 -> 32 genetic alignment network.

    Architecture:

        128 -> 64 -> 32
    """

    def __init__(
        self,
        input_dimension: int = 128,
        output_dimension: int = 32,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dimension),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


# ---------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------

class GeneticService:

    def __init__(self) -> None:

        self.transformer = None
        self.alignment = None

        self.feature_columns = None
        self.target_column = "Target"

        self.input_dim = None
        self.embedding_dim = None

        self.alignment_input_dim = None
        self.alignment_output_dim = None

        self._load_models()

    # -----------------------------------------------------------------
    # Load frozen models
    # -----------------------------------------------------------------

    def _load_models(self) -> None:

        if not TRANSFORMER_PATH.exists():
            raise FileNotFoundError(
                f"Genetic Transformer not found:\n"
                f"{TRANSFORMER_PATH}"
            )

        if not ALIGNMENT_PATH.exists():
            raise FileNotFoundError(
                f"Genetic alignment model not found:\n"
                f"{ALIGNMENT_PATH}"
            )

        # -------------------------------------------------------------
        # Transformer checkpoint
        # -------------------------------------------------------------

        transformer_state = torch.load(
            TRANSFORMER_PATH,
            map_location="cpu",
            weights_only=False,
        )

        self.transformer = GeneticTransformer()

        self.transformer.load_state_dict(
            transformer_state
        )

        self.transformer.eval()

        self.input_dim = 1000
        self.embedding_dim = 128

        # -------------------------------------------------------------
        # Alignment checkpoint
        # -------------------------------------------------------------

        alignment_state = torch.load(
            ALIGNMENT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        self.alignment = GeneticAlignment(
            input_dimension=128,
            output_dimension=32,
        )

        self.alignment.load_state_dict(
            alignment_state
        )

        self.alignment.eval()

        self.alignment_input_dim = 128
        self.alignment_output_dim = 32

    # -----------------------------------------------------------------
    # Input preparation
    # -----------------------------------------------------------------

    def prepare_features(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:

        if dataframe is None or dataframe.empty:
            raise ValueError(
                "Genetic input data is empty."
            )

        # ------------------------------------------------------------------
        # METADATA COLUMNS
        # ------------------------------------------------------------------
        #
        # Target is ground-truth information and is NOT required during
        # single-patient inference.
        #
        metadata_columns = [
            "GSM_ID",
            "Source",
            "Subtype",
            "Target",
        ]

        # ------------------------------------------------------------------
        # FIND ACTUAL GENETIC FEATURES
        # ------------------------------------------------------------------

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in metadata_columns
        ]

        if len(feature_columns) != 1000:
            raise ValueError(
                "Expected exactly 1000 genetic features, "
                f"found {len(feature_columns)}."
            )

        # ------------------------------------------------------------------
        # NUMERIC CONVERSION
        # ------------------------------------------------------------------

        features = dataframe[
            feature_columns
        ].copy()

        features = features.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if features.isnull().any().any():
            raise ValueError(
                "Genetic input contains missing or "
                "non-numeric genetic feature values."
            )

        self.feature_columns = feature_columns

        return features.to_numpy(
            dtype=np.float32
        )

    # -----------------------------------------------------------------
    # 128-D representation
    # -----------------------------------------------------------------

    def encode(
        self,
        features: np.ndarray,
    ) -> np.ndarray:

        if self.transformer is None:
            raise RuntimeError(
                "Genetic Transformer is not loaded."
            )

        tensor = torch.tensor(
            features,
            dtype=torch.float32,
        )

        with torch.no_grad():

            embedding = self.transformer(
                tensor
            )

        return embedding.cpu().numpy()

    # -----------------------------------------------------------------
    # 32-D alignment
    # -----------------------------------------------------------------

    def align(
        self,
        embedding_128: np.ndarray,
    ) -> np.ndarray:

        if self.alignment is None:
            raise RuntimeError(
                "Genetic alignment model is not loaded."
            )

        tensor = torch.tensor(
            embedding_128,
            dtype=torch.float32,
        )

        with torch.no_grad():

            aligned = self.alignment(
                tensor
            )

        return aligned.cpu().numpy()

    # -----------------------------------------------------------------
    # Complete representation pipeline
    # -----------------------------------------------------------------

    def process(
        self,
        input_data: pd.DataFrame,
    ) -> Dict[str, Any]:

        features = self.prepare_features(
            input_data
        )

        embedding_128 = self.encode(
            features
        )

        embedding_32 = self.align(
            embedding_128
        )

        return {
            "modality": "genetic",

            "input_rows": int(
                len(input_data)
            ),

            "input_features": int(
                features.shape[1]
            ),

            "embedding_dimension": int(
                embedding_128.shape[1]
            ),

            "aligned_dimension": int(
                embedding_32.shape[1]
            ),

            "embedding_128": (
                embedding_128.tolist()
            ),

            "aligned_embedding_32": (
                embedding_32.tolist()
            ),
        }


__all__ = [
    "GeneticService",
]