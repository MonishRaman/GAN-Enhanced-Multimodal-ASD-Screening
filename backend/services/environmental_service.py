"""
Environmental Model Service
===========================

Application service for the frozen Environmental ASD model.

Pipeline:

    environmental input
            ↓
    exact trained environmental features
            ↓
    frozen environmental encoder
            ↓
    128-dimensional representation
            ↓
    frozen 128 → 32 alignment
            ↓
    32-dimensional environmental representation
            ↓
    multimodal fusion

This service does NOT train or modify any model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backend.utils.paths import ENVIRONMENTAL_PREPROCESSOR


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

SERVICE_FILE = Path(__file__).resolve()

BACKEND_DIR = SERVICE_FILE.parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

ENVIRONMENTAL_MODEL_DIR = PROJECT_ROOT / "Environmental_Model"

ENCODER_PATH = (
    ENVIRONMENTAL_MODEL_DIR
    / "03_representation_learning"
    / "environmental_encoder_128d.pt"
)

ALIGNMENT_PATH = (
    ENVIRONMENTAL_MODEL_DIR
    / "04_cross_modal_alignment"
    / "environmental_128_to_32_alignment.pt"
)


# ---------------------------------------------------------------------
# Neural network definitions
# ---------------------------------------------------------------------

class EnvironmentalEncoder(nn.Module):
    """
    Exact architecture required by the frozen environmental
    representation checkpoint.

    Checkpoint structure:

        encoder.0.weight        -> Linear(31, 128)
        encoder.0.bias
        encoder.1.*             -> BatchNorm1d(128)
        encoder.4.weight        -> Linear(128, 128)
        encoder.4.bias
        encoder.5.*             -> BatchNorm1d(128)
        classifier.weight       -> Linear(128, 1)
        classifier.bias

    The classifier is retained so the complete checkpoint can be
    loaded correctly.

    For multimodal fusion, only the 128-dimensional encoder
    representation is used.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the classifier output.
        """

        representation = self.encoder(x)

        return self.classifier(
            representation
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the 128-dimensional frozen representation.
        """

        return self.encoder(x)


class EnvironmentalAlignment(nn.Module):
    """
    Exact 128 → 32 environmental alignment architecture.

    Encoder:

        128 → 64 → 32

    Decoder:

        32 → 64 → 128

    BatchNorm layers are included because they are present
    in the trained checkpoint state dictionary.
    """

    def __init__(
        self,
        input_dimension: int = 128,
        output_dimension: int = 32,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dimension,
                64,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(
                64,
                output_dimension,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                output_dimension,
                64,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(
                64,
                input_dimension,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the 32-dimensional aligned representation.
        """

        return self.encoder(x)

    def reconstruct(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct a 128-dimensional representation.
        """

        aligned = self.encoder(x)

        return self.decoder(aligned)


# ---------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------

class EnvironmentalService:
    """
    Loads and runs the frozen Environmental model pipeline.
    """

    def __init__(self) -> None:

        self.preprocessor: Any = None
        self.encoder: EnvironmentalEncoder | None = None
        self.alignment: EnvironmentalAlignment | None = None

        self.feature_columns: list[str] | None = None
        self.target_column: str | None = None

        self.normalization_mean: Any = None
        self.normalization_std: Any = None

        self.input_dim: int | None = None
        self.embedding_dim: int | None = None

        self.alignment_input_dim: int | None = None
        self.alignment_output_dim: int | None = None

        self._load_models()

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def _load_models(self) -> None:
        """
        Load the frozen environmental encoder and alignment model.
        """

        if not ENCODER_PATH.exists():
            raise FileNotFoundError(
                "Environmental encoder not found:\n"
                f"{ENCODER_PATH}"
            )

        if not ALIGNMENT_PATH.exists():
            raise FileNotFoundError(
                "Environmental alignment model not found:\n"
                f"{ALIGNMENT_PATH}"
            )

        # -------------------------------------------------------------
        # Load frozen preprocessor first so its metadata is authoritative
        # -------------------------------------------------------------

        self.preprocessor = joblib.load(
            ENVIRONMENTAL_PREPROCESSOR
        )

        if self.preprocessor is None:
            raise RuntimeError(
                "Environmental preprocessor could not be loaded."
            )

        # -------------------------------------------------------------
        # Load checkpoints
        # -------------------------------------------------------------

        encoder_checkpoint = torch.load(
            ENCODER_PATH,
            map_location="cpu",
            weights_only=False,
        )

        alignment_checkpoint = torch.load(
            ALIGNMENT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        # -------------------------------------------------------------
        # Validate encoder checkpoint
        # -------------------------------------------------------------

        required_encoder_keys = {
            "model_state_dict",
            "input_dim",
            "embedding_dim",
            "feature_columns",
            "target_column",
        }

        missing_encoder_keys = (
            required_encoder_keys
            - set(encoder_checkpoint.keys())
        )

        if missing_encoder_keys:
            raise RuntimeError(
                "Environmental encoder checkpoint is missing "
                f"required metadata: {sorted(missing_encoder_keys)}"
            )

        # -------------------------------------------------------------
        # Read encoder metadata
        # -------------------------------------------------------------

        self.input_dim = int(
            encoder_checkpoint["input_dim"]
        )

        self.embedding_dim = int(
            encoder_checkpoint["embedding_dim"]
        )

        if hasattr(self.preprocessor, "feature_names_in_"):
            preprocessor_columns = list(
                self.preprocessor.feature_names_in_
            )

            if not preprocessor_columns:
                raise RuntimeError(
                    "Environmental preprocessor feature_names_in_ is empty."
                )

            self.feature_columns = preprocessor_columns

            if self.input_dim != len(self.feature_columns):
                raise RuntimeError(
                    "Environmental preprocessor feature count mismatch: "
                    f"input_dim={self.input_dim}, "
                    f"preprocessor feature count={len(self.feature_columns)}"
                )

        else:
            self.feature_columns = list(
                encoder_checkpoint["feature_columns"]
            )

        self.target_column = str(
            encoder_checkpoint["target_column"]
        )

        # -------------------------------------------------------------
        # Safety validation
        # -------------------------------------------------------------

        if self.input_dim != len(self.feature_columns):
            raise RuntimeError(
                "Environmental checkpoint inconsistency: "
                f"input_dim={self.input_dim}, "
                f"but feature_columns={len(self.feature_columns)}"
            )

        if self.embedding_dim != 128:
            raise RuntimeError(
                "Unexpected environmental embedding dimension: "
                f"{self.embedding_dim}. Expected 128."
            )

        # -------------------------------------------------------------
        # Build exact encoder architecture
        # -------------------------------------------------------------

        self.encoder = EnvironmentalEncoder(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
        )

        try:
            self.encoder.load_state_dict(
                encoder_checkpoint["model_state_dict"],
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Environmental encoder checkpoint does not match "
                "the service architecture.\n"
                f"Checkpoint: {ENCODER_PATH}\n\n"
                f"{exc}"
            ) from exc

        self.encoder.eval()

        # -------------------------------------------------------------
        # Validate alignment checkpoint
        # -------------------------------------------------------------

        required_alignment_keys = {
            "model_state_dict",
            "input_dimension",
            "output_dimension",
        }

        missing_alignment_keys = (
            required_alignment_keys
            - set(alignment_checkpoint.keys())
        )

        if missing_alignment_keys:
            raise RuntimeError(
                "Environmental alignment checkpoint is missing "
                f"required metadata: {sorted(missing_alignment_keys)}"
            )

        # -------------------------------------------------------------
        # Read alignment metadata
        # -------------------------------------------------------------

        self.alignment_input_dim = int(
            alignment_checkpoint["input_dimension"]
        )

        self.alignment_output_dim = int(
            alignment_checkpoint["output_dimension"]
        )

        self.normalization_mean = (
            alignment_checkpoint.get(
                "normalization_mean"
            )
        )

        self.normalization_std = (
            alignment_checkpoint.get(
                "normalization_std"
            )
        )

        if self.alignment_input_dim != 128:
            raise RuntimeError(
                "Unexpected environmental alignment input dimension: "
                f"{self.alignment_input_dim}. Expected 128."
            )

        if self.alignment_output_dim != 32:
            raise RuntimeError(
                "Unexpected environmental alignment output dimension: "
                f"{self.alignment_output_dim}. Expected 32."
            )

        # -------------------------------------------------------------
        # Build exact alignment architecture
        # -------------------------------------------------------------

        self.alignment = EnvironmentalAlignment(
            input_dimension=self.alignment_input_dim,
            output_dimension=self.alignment_output_dim,
        )

        try:
            self.alignment.load_state_dict(
                alignment_checkpoint["model_state_dict"],
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Environmental alignment checkpoint does not "
                "match the service architecture.\n"
                f"Checkpoint: {ALIGNMENT_PATH}\n\n"
                f"{exc}"
            ) from exc

        self.alignment.eval()

    # -----------------------------------------------------------------
    # CSV loading
    # -----------------------------------------------------------------

    def load_input(
        self,
        csv_path: str | Path,
    ) -> pd.DataFrame:
        """
        Load an environmental input CSV.
        """

        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Environmental input file not found:\n{path}"
            )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(
                "Environmental input CSV is empty."
            )

        return dataframe

    # -----------------------------------------------------------------
    # Feature preparation
    # -----------------------------------------------------------------

    def prepare_features(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Extract the exact feature columns used by the trained
        environmental encoder.
        """

        if self.feature_columns is None:
            raise RuntimeError(
                "Environmental feature columns are not loaded."
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Environmental input is missing required "
                f"feature columns: {missing_columns}"
            )

        features = dataframe[
            self.feature_columns
        ].copy()

        # -------------------------------------------------------------
        # Convert required features to numeric
        # -------------------------------------------------------------

        features = features.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Missing numeric values are intentionally allowed here.
        # The frozen environmental preprocessing pipeline contains:
        #
        #     SimpleImputer(strategy="median")
        #     -> StandardScaler()
        #
        # Therefore, missing values must be passed through to the
        # frozen preprocessor instead of being rejected at the service layer.

        if self.input_dim is not None:
            if features.shape[1] != self.input_dim:
                raise RuntimeError(
                    "Prepared environmental feature dimension "
                    f"is {features.shape[1]}, but the trained "
                    f"encoder expects {self.input_dim}."
                )

        return features.to_numpy(
            dtype=np.float32
        )

    # -----------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------

    def encode(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Generate the frozen 128-dimensional environmental
        representation.

        The classifier head is NOT used here because the purpose
        of this stage is to provide the representation for
        multimodal fusion.
        """

        if self.encoder is None:
            raise RuntimeError(
                "Environmental encoder is not loaded."
            )

        tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
        )

        if tensor.ndim != 2:
            raise ValueError(
                "Environmental features must be a 2-D array."
            )

        if self.input_dim is not None:
            if tensor.shape[1] != self.input_dim:
                raise ValueError(
                    "Environmental feature dimension mismatch: "
                    f"received {tensor.shape[1]}, "
                    f"expected {self.input_dim}."
                )

        with torch.no_grad():
            embedding = self.encoder.encode(
                tensor
            )

        return embedding.cpu().numpy()

    # -----------------------------------------------------------------
    # Alignment
    # -----------------------------------------------------------------

    def align(
        self,
        embedding_128: np.ndarray,
    ) -> np.ndarray:
        """
        Convert the environmental 128-dimensional representation
        into the frozen 32-dimensional aligned representation.
        """

        if self.alignment is None:
            raise RuntimeError(
                "Environmental alignment model is not loaded."
            )

        tensor = torch.as_tensor(
            embedding_128,
            dtype=torch.float32,
        )

        if tensor.ndim != 2:
            raise ValueError(
                "Environmental 128-D embedding must be a "
                "2-D array."
            )

        if tensor.shape[1] != 128:
            raise ValueError(
                "Environmental embedding dimension mismatch: "
                f"received {tensor.shape[1]}, expected 128."
            )

        # -------------------------------------------------------------
        # Apply normalization saved with the alignment checkpoint
        # -------------------------------------------------------------

        if (
            self.normalization_mean is not None
            and self.normalization_std is not None
        ):

            mean = torch.as_tensor(
                self.normalization_mean,
                dtype=torch.float32,
            )

            std = torch.as_tensor(
                self.normalization_std,
                dtype=torch.float32,
            )

            # Support either scalar or 128-D normalization values.
            if mean.numel() not in (1, 128):
                raise RuntimeError(
                    "Unexpected environmental alignment "
                    "normalization_mean size: "
                    f"{mean.numel()}"
                )

            if std.numel() not in (1, 128):
                raise RuntimeError(
                    "Unexpected environmental alignment "
                    "normalization_std size: "
                    f"{std.numel()}"
                )

            # Prevent division by zero.
            std = torch.where(
                std == 0,
                torch.ones_like(std),
                std,
            )

            tensor = (
                tensor - mean
            ) / std

        with torch.no_grad():
            aligned = self.alignment(
                tensor
            )

        return aligned.cpu().numpy()

    # -----------------------------------------------------------------
    # Complete pipeline
    # -----------------------------------------------------------------

    def process(
        self,
        csv_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Run the complete frozen Environmental pipeline.
        """

        dataframe = self.load_input(
            csv_path
        )

        if self.preprocessor is None:
            raise RuntimeError(
                "Environmental preprocessor is not loaded."
            )

        if self.feature_columns is None:
            raise RuntimeError(
                "Environmental feature columns are not loaded."
            )

        exact_features = dataframe[
            self.feature_columns
        ].copy()

        # -------------------------------------------------------------
        # STEP 1 — FROZEN ENVIRONMENTAL PREPROCESSING
        #
        # The original preprocessing artifact was created with
        # scikit-learn 1.6.1. The current runtime uses 1.9.x,
        # where the pickled SimpleImputer is not directly executable.
        #
        # Reproduce the frozen pipeline mathematically using the
        # parameters learned by that artifact:
        #
        #     median imputation
        #          ↓
        #     StandardScaler
        #
        # No model parameters are changed.
        # -------------------------------------------------------------

        raw_features = exact_features.to_numpy(
            dtype=np.float64
        )

        # Frozen SimpleImputer(strategy="median")
        imputer = self.preprocessor.named_steps["imputer"]

        if not hasattr(imputer, "statistics_"):
            raise RuntimeError(
                "Frozen environmental imputer does not contain "
                "learned statistics."
            )

        imputer_statistics = np.asarray(
            imputer.statistics_,
            dtype=np.float64,
        )

        if imputer_statistics.shape[0] != raw_features.shape[1]:
            raise RuntimeError(
                "Environmental imputer dimension mismatch: "
                f"statistics={imputer_statistics.shape[0]}, "
                f"features={raw_features.shape[1]}"
            )

        missing_mask = np.isnan(raw_features)

        if missing_mask.any():
            raw_features = raw_features.copy()

            rows, columns = np.where(
                missing_mask
            )

            raw_features[
                rows,
                columns
            ] = imputer_statistics[
                columns
            ]

        # Frozen StandardScaler
        scaler = self.preprocessor.named_steps["scaler"]

        if not hasattr(scaler, "mean_") or not hasattr(
            scaler,
            "scale_",
        ):
            raise RuntimeError(
                "Frozen environmental scaler does not contain "
                "learned parameters."
            )

        scaler_mean = np.asarray(
            scaler.mean_,
            dtype=np.float64,
        )

        scaler_scale = np.asarray(
            scaler.scale_,
            dtype=np.float64,
        )

        if (
            scaler_mean.shape[0]
            != raw_features.shape[1]
        ):
            raise RuntimeError(
                "Environmental scaler mean dimension mismatch."
            )

        if (
            scaler_scale.shape[0]
            != raw_features.shape[1]
        ):
            raise RuntimeError(
                "Environmental scaler scale dimension mismatch."
            )

        if np.any(scaler_scale == 0):
            raise RuntimeError(
                "Environmental scaler contains zero scale."
            )

        processed = (
            raw_features - scaler_mean
        ) / scaler_scale

        processed = np.asarray(
            processed,
            dtype=np.float32,
        )

        if not np.isfinite(processed).all():
            raise RuntimeError(
                "Environmental preprocessing produced "
                "non-finite values."
            )

        if processed.ndim == 1:
            processed = processed.reshape(
                1,
                -1,
            )

        embedding_128 = self.encode(
            processed
        )

        embedding_32 = self.align(
            embedding_128
        )

        features = processed

        return {
            "modality": "environmental",

            "input_rows": int(
                len(dataframe)
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


# ---------------------------------------------------------------------
# Shared service instance
# ---------------------------------------------------------------------

environmental_service = EnvironmentalService()


__all__ = [
    "EnvironmentalService",
    "environmental_service",
]