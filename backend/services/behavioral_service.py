"""
Behavioral ASD inference service.

This service runs the FROZEN behavioral model pipeline.

Pipeline:
    Raw behavioral input
        ↓
    Frozen behavioral preprocessor
        ↓
    87-dimensional representation
        ↓
    Frozen behavioral encoder
        ↓
    128-dimensional embedding
        ↓
    Frozen 128 → 32 alignment model
        ↓
    32-dimensional aligned representation
        ↓
    Frozen behavioral scaler
        ↓
    Frozen behavioral expert
        ↓
    ASD probability
        ↓
    ASD / NON-ASD prediction

IMPORTANT:
    - Inference only.
    - No training.
    - No fitting.
    - No modification of model artifacts.
    - No replacement of the frozen model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backend.utils.paths import PROJECT_ROOT


# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# BEHAVIORAL MODEL DIMENSIONS
# ---------------------------------------------------------------------------

BEHAVIORAL_INPUT_DIM = 87
BEHAVIORAL_EMBEDDING_DIM = 128
BEHAVIORAL_ALIGNMENT_INPUT_DIM = 128
BEHAVIORAL_ALIGNMENT_OUTPUT_DIM = 32


# ---------------------------------------------------------------------------
# SKLEARN COMPATIBILITY
# ---------------------------------------------------------------------------

def _install_sklearn_compatibility_shim() -> None:
    """
    Restore the private sklearn class name required to unpickle the
    behavioral ColumnTransformer.

    The behavioral preprocessor was saved with scikit-learn 1.6.1.

    Current environment:
        scikit-learn 1.9.x

    scikit-learn versions newer than 1.6.1 removed the private
    _RemainderColsList class used inside older ColumnTransformer
    pickle files.

    We restore that class name only for unpickling.

    IMPORTANT:
        This does NOT retrain, refit, or modify the saved preprocessor.
        It only provides the class name expected by the pickle.
    """

    try:
        import sklearn.compose._column_transformer as column_transformer

        if not hasattr(
            column_transformer,
            "_RemainderColsList",
        ):
            class _RemainderColsList(list):
                """
                Compatibility representation for sklearn 1.6.1
                ColumnTransformer remainder columns.
                """

                pass

            column_transformer._RemainderColsList = (
                _RemainderColsList
            )

    except Exception as exc:
        raise RuntimeError(
            "Unable to install scikit-learn compatibility shim "
            "required by the frozen behavioral preprocessor."
        ) from exc


# ---------------------------------------------------------------------------
# EXACT FROZEN BEHAVIORAL NETWORK ARCHITECTURES
# ---------------------------------------------------------------------------

class BehavioralEncoder(nn.Module):
    """
    Exact architecture of the saved behavioral encoder.

    Input:
        87 processed behavioral features

    Representation:
        87
        ↓
        Linear 87 → 128
        ↓
        BatchNorm
        ↓
        ReLU
        ↓
        Dropout
        ↓
        Linear 128 → 128
        ↓
        BatchNorm
        ↓
        ReLU
        ↓
        Dropout
        ↓
        128-dimensional embedding

    IMPORTANT:
        The saved checkpoint also contains:
            classifier.weight
            classifier.bias

        Those belong to the original training classifier and are NOT
        part of the 128-D representation used by the multimodal pipeline.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(87, 128),      # encoder.0
            nn.BatchNorm1d(128),     # encoder.1
            nn.ReLU(),              # encoder.2
            nn.Dropout(0.2),        # encoder.3

            nn.Linear(128, 128),    # encoder.4
            nn.BatchNorm1d(128),    # encoder.5
            nn.ReLU(),              # encoder.6
            nn.Dropout(0.2),        # encoder.7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------

class BehavioralAlignment(nn.Module):
    """
    Exact behavioral cross-modal alignment architecture.

    The saved checkpoint is an autoencoder:

        ENCODER
        128
         ↓
        64
         ↓
        32

        DECODER
        32
         ↓
        64
         ↓
        128

    For multimodal inference, only the 32-D encoder representation
    is passed downstream.

    IMPORTANT:
        The decoder is still defined because its weights are present
        in the frozen checkpoint and strict checkpoint loading must
        succeed.
    """

    def __init__(self) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # 128 → 64 → 32 ENCODER
        # ---------------------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # ---------------------------------------------------------------
        # 32 → 64 → 128 DECODER
        # ---------------------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 32-dimensional aligned behavioral representation.

        The decoder is intentionally not used during application
        inference because downstream behavioral fusion expects 32-D.
        """

        return self.encoder(x)


# ---------------------------------------------------------------------------

class BehavioralExpert(nn.Module):
    """
    Frozen behavioral expert.

    Architecture:

        32
        ↓
        64
        ↓ BatchNorm
        ↓ ReLU
        ↓ Dropout
        ↓
        32
        ↓ ReLU
        ↓ Dropout
        ↓
        1
    """

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


# ---------------------------------------------------------------------------
# BEHAVIORAL SERVICE
# ---------------------------------------------------------------------------

class BehavioralService:
    """
    Production inference wrapper for the frozen behavioral ASD model.
    """

    def __init__(self) -> None:

        self.project_root = Path(PROJECT_ROOT)

        # ---------------------------------------------------------------
        # VERIFIED DIMENSIONS
        # ---------------------------------------------------------------

        self.input_dim = BEHAVIORAL_INPUT_DIM
        self.embedding_dim = BEHAVIORAL_EMBEDDING_DIM
        self.alignment_input_dim = (
            BEHAVIORAL_ALIGNMENT_INPUT_DIM
        )
        self.alignment_output_dim = (
            BEHAVIORAL_ALIGNMENT_OUTPUT_DIM
        )

        # ---------------------------------------------------------------
        # FEATURE NAMES
        # ---------------------------------------------------------------

        self.feature_names_path = (
            self.project_root
            / "Behavioral_Model"
            / "02_preprocessing"
            / "behavioral_processed_feature_names.txt"
        )

        self.feature_columns = self._load_feature_columns()

        if len(self.feature_columns) != self.input_dim:
            raise RuntimeError(
                "Behavioral feature metadata mismatch. "
                f"Expected {self.input_dim} features, "
                f"found {len(self.feature_columns)}."
            )

        # ---------------------------------------------------------------
        # VERIFIED ARTIFACT LOCATIONS
        # ---------------------------------------------------------------

        self.preprocessor_path = (
            self.project_root
            / "Behavioral_Model"
            / "02_preprocessing"
            / "behavioral_preprocessor.joblib"
        )

        self.encoder_path = (
            self.project_root
            / "Behavioral_Model"
            / "03_representation_learning"
            / "behavioral_encoder_128d.pt"
        )

        self.alignment_path = (
            self.project_root
            / "Behavioral_Model"
            / "04_cross_modal_alignment"
            / "behavioral_128_to_32_alignment.pt"
        )

        self.scaler_path = (
            self.project_root
            / "05_multimodal_fusion"
            / "M3_1_behavioral_scaler.joblib"
        )

        self.expert_path = (
            self.project_root
            / "05_multimodal_fusion"
            / "M3_1_behavioral_expert.pt"
        )

        # ---------------------------------------------------------------
        # LOAD FROZEN ARTIFACTS
        # ---------------------------------------------------------------

        self.preprocessor = self._load_joblib(
            self.preprocessor_path,
            "behavioral preprocessor",
        )

        self.scaler = self._load_joblib(
            self.scaler_path,
            "behavioral scaler",
        )

        self.encoder = self._load_model(
            BehavioralEncoder(),
            self.encoder_path,
            "behavioral encoder",
        )

        self.alignment = self._load_model(
            BehavioralAlignment(),
            self.alignment_path,
            "behavioral alignment",
        )

        self.expert = self._load_model(
            BehavioralExpert(),
            self.expert_path,
            "behavioral expert",
        )

    # -------------------------------------------------------------------
    # FEATURE METADATA
    # -------------------------------------------------------------------

    def _load_feature_columns(self) -> list[str]:
        """
        Load the exact 87 processed feature names produced during
        behavioral preprocessing.
        """

        if not self.feature_names_path.exists():
            raise FileNotFoundError(
                "Behavioral processed feature names file not found:\n"
                f"{self.feature_names_path}"
            )

        columns: list[str] = []

        with self.feature_names_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                # File entries are formatted like:
                #
                # 1. numeric__A1_Score
                #
                # Remove the numeric prefix only.
                if ". " in line:
                    _, feature_name = line.split(
                        ". ",
                        1,
                    )
                else:
                    feature_name = line

                columns.append(feature_name.strip())

        return columns

    # -------------------------------------------------------------------
    # JOBLIB LOADER
    # -------------------------------------------------------------------

    @staticmethod
    def _load_joblib(
        path: Path,
        name: str,
    ) -> Any:
        """
        Load a frozen sklearn/joblib artifact.

        A compatibility shim is installed because the behavioral
        ColumnTransformer was serialized using sklearn 1.6.1 while
        the current environment uses a newer sklearn version.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

        # Install compatibility before joblib attempts to unpickle
        # the old ColumnTransformer.
        _install_sklearn_compatibility_shim()

        try:
            return joblib.load(path)

        except AttributeError as exc:

            message = str(exc)

            if "_RemainderColsList" in message:

                # Retry once after explicitly installing the shim.
                _install_sklearn_compatibility_shim()

                try:
                    return joblib.load(path)

                except Exception as retry_exc:
                    raise RuntimeError(
                        f"Unable to load {name} even after applying "
                        "the sklearn 1.6.1 compatibility shim.\n"
                        f"Artifact: {path}\n"
                        f"Original error: {retry_exc}"
                    ) from retry_exc

            raise RuntimeError(
                f"Unable to load {name}.\n"
                f"Artifact: {path}\n"
                f"Error: {exc}"
            ) from exc

    # -------------------------------------------------------------------
    # PYTORCH CHECKPOINT HANDLING
    # -------------------------------------------------------------------

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract a PyTorch state_dict from common checkpoint formats.
        """

        if isinstance(checkpoint, dict):

            # Direct state_dict
            if checkpoint and all(
                isinstance(value, torch.Tensor)
                for value in checkpoint.values()
            ):
                return checkpoint

            # Wrapped state_dict
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "network_state_dict",
            ):

                value = checkpoint.get(key)

                if isinstance(value, dict):

                    if value and all(
                        isinstance(v, torch.Tensor)
                        for v in value.values()
                    ):
                        return value

        raise RuntimeError(
            "Unable to identify a valid PyTorch state_dict."
        )

    # -------------------------------------------------------------------

    @classmethod
    def _load_model(
        cls,
        model: nn.Module,
        path: Path,
        name: str,
    ) -> nn.Module:
        """
        Load a frozen PyTorch checkpoint.

        Strict loading is intentional. If the checkpoint does not
        match the verified architecture, inference must stop.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found:\n{path}"
            )

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

        state_dict = cls._extract_state_dict(
            checkpoint
        )

        cleaned_state_dict: Dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key[len("module."):]

            cleaned_state_dict[key] = value

        # The behavioral encoder checkpoint also contains the original
        # training classifier head. The multimodal application does not
        # use that classifier; it uses only the 128-D encoder embedding.
        #
        # Therefore remove only the classifier parameters before loading
        # the frozen representation encoder.
        if isinstance(model, BehavioralEncoder):
            cleaned_state_dict = {
                key: value
                for key, value in cleaned_state_dict.items()
                if not key.startswith("classifier.")
            }

        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        model.to(DEVICE)
        model.eval()

        return model

    # -------------------------------------------------------------------
    # INPUT PREPARATION
    # -------------------------------------------------------------------

    @staticmethod
    def _prepare_input(
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare raw behavioral input.

        The training dataset contains:

            Class/ASD

        as the target.

        The target must never enter inference.
        """

        if not isinstance(
            input_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "Behavioral input must be a pandas DataFrame."
            )

        if input_data.empty:
            raise ValueError(
                "Behavioral input DataFrame is empty."
            )

        data = input_data.copy()

        # ---------------------------------------------------------------
        # NEVER USE TARGET DURING INFERENCE
        # ---------------------------------------------------------------

        target_columns = [
            "Class/ASD",
            "class/ASD",
            "Class",
            "ASD",
        ]

        for column in target_columns:

            if column in data.columns:
                data = data.drop(
                    columns=[column]
                )

        # ---------------------------------------------------------------
        # REMOVE LEAKAGE COLUMN
        # ---------------------------------------------------------------

        # The preprocessing report explicitly states that "result"
        # was removed as leakage during model development.
        if "result" in data.columns:
            data = data.drop(
                columns=["result"]
            )

        return data

    # -------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        input_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run the complete frozen behavioral inference pipeline.

        Returns:

            {
                "probability": float,
                "prediction": "ASD" | "NON-ASD",
                "prediction_numeric": 0 | 1
            }
        """

        # ---------------------------------------------------------------
        # PREPARE RAW INPUT
        # ---------------------------------------------------------------

        data = self._prepare_input(
            input_data
        )

        # ---------------------------------------------------------------
        # STEP 1
        # FROZEN BEHAVIORAL PREPROCESSOR
        # ---------------------------------------------------------------

        processed = self.preprocessor.transform(
            data
        )

        if hasattr(
            processed,
            "toarray",
        ):
            processed = processed.toarray()

        processed = np.asarray(
            processed,
            dtype=np.float32,
        )

        if processed.ndim == 1:
            processed = processed.reshape(
                1,
                -1,
            )

        if processed.ndim != 2:
            raise ValueError(
                "Behavioral preprocessor returned an "
                f"invalid shape: {processed.shape}"
            )

        if processed.shape[1] != self.input_dim:
            raise ValueError(
                "Unexpected behavioral preprocessor "
                "output dimension. "
                f"Expected {self.input_dim}, "
                f"received {processed.shape[1]}."
            )

        # ---------------------------------------------------------------
        # STEP 2
        # 87-D → 128-D BEHAVIORAL ENCODER
        # ---------------------------------------------------------------

        processed_tensor = torch.tensor(
            processed,
            dtype=torch.float32,
            device=DEVICE,
        )

        # ---------------------------------------------------------------
        # STEP 2 — FROZEN BEHAVIORAL ENCODER
        # ---------------------------------------------------------------

        embedding_128 = self.encoder(
            processed_tensor
        )

        embedding_128 = torch.as_tensor(
            embedding_128,
            dtype=torch.float32,
            device=DEVICE,
        )

        if embedding_128.ndim != 2:
            raise ValueError(
                "Behavioral encoder returned an "
                f"invalid shape: {tuple(embedding_128.shape)}"
            )

        if (
            embedding_128.shape[1]
            != self.embedding_dim
        ):
            raise ValueError(
                "Unexpected behavioral embedding "
                "dimension. "
                f"Expected {self.embedding_dim}, "
                f"received {embedding_128.shape[1]}."
            )

        # ---------------------------------------------------------------
        # STEP 3
        # 128-D → 32-D ALIGNMENT
        # ---------------------------------------------------------------

        aligned_32 = self.alignment(
            embedding_128
        )

        if aligned_32.ndim != 2:
            raise ValueError(
                "Behavioral alignment returned an "
                f"invalid shape: {tuple(aligned_32.shape)}"
            )

        if (
            aligned_32.shape[1]
            != self.alignment_output_dim
        ):
            raise ValueError(
                "Unexpected behavioral aligned "
                "dimension. "
                f"Expected {self.alignment_output_dim}, "
                f"received {aligned_32.shape[1]}."
            )

        # ---------------------------------------------------------------
        # STEP 4
        # FROZEN BEHAVIORAL SCALER
        # ---------------------------------------------------------------

        aligned_numpy = (
            aligned_32
            .cpu()
            .numpy()
        )

        scaled_32 = self.scaler.transform(
            aligned_numpy
        )

        scaled_32 = np.asarray(
            scaled_32,
            dtype=np.float32,
        )

        if scaled_32.ndim != 2:
            raise ValueError(
                "Behavioral scaler returned an "
                f"invalid shape: {scaled_32.shape}"
            )

        if (
            scaled_32.shape[1]
            != self.alignment_output_dim
        ):
            raise ValueError(
                "Unexpected behavioral scaler "
                "output dimension. "
                f"Expected {self.alignment_output_dim}, "
                f"received {scaled_32.shape[1]}."
            )

        # ---------------------------------------------------------------
        # STEP 5
        # FROZEN BEHAVIORAL EXPERT
        # ---------------------------------------------------------------

        expert_input = torch.tensor(
            scaled_32,
            dtype=torch.float32,
            device=DEVICE,
        )

        logits = self.expert(
            expert_input
        )

        if logits.ndim == 1:
            logits = logits.reshape(
                -1,
                1,
            )

        if logits.shape[1] != 1:
            raise ValueError(
                "Behavioral expert returned "
                f"unexpected output shape: {tuple(logits.shape)}"
            )

        # ---------------------------------------------------------------
        # STEP 6
        # SIGMOID PROBABILITY
        # ---------------------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy().reshape(-1)

        if len(probabilities) == 0:
            raise RuntimeError(
                "Behavioral expert returned no probability."
            )

        probability = float(
            probabilities[0]
        )

        # ---------------------------------------------------------------
        # STEP 7
        # STANDARD RESULT FORMAT
        # ---------------------------------------------------------------

        return self._format_result(
            probability
        )

    # -------------------------------------------------------------------
    # RESULT FORMAT
    # -------------------------------------------------------------------

    @staticmethod
    def _format_result(
        probability: float,
    ) -> Dict[str, Any]:
        """
        Convert probability into the project's standard result format.

        Verified classification threshold:
            0.5
        """

        probability = float(
            probability
        )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Invalid behavioral probability: "
                f"{probability}"
            )

        prediction_numeric = int(
            probability >= 0.5
        )

        return {
            "probability": probability,
            "prediction": (
                "ASD"
                if prediction_numeric == 1
                else "NON-ASD"
            ),
            "prediction_numeric": prediction_numeric,
        }