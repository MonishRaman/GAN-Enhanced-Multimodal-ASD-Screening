"""
ASD Multimodal Project
Inference Result Schemas
------------------------

Defines the response structures returned by the backend inference API.

These schemas describe results from:

    Genetic
    Behavioral
    Environmental
    Genetic + Behavioral
    Genetic + Environmental
    Behavioral + Environmental
    Genetic + Behavioral + Environmental

This module contains validation/representation logic only.

It does NOT:
    - load models
    - perform preprocessing
    - perform inference
    - calculate fusion weights
    - modify frozen artifacts
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

PredictionLabel = Literal[
    "ASD",
    "NON-ASD",
]

ModalityName = Literal[
    "genetic",
    "behavioral",
    "environmental",
]

InferenceMode = Literal[
    "genetic",
    "behavioral",
    "environmental",
    "genetic_behavioral",
    "genetic_environmental",
    "behavioral_environmental",
    "all_three",
]


# ============================================================================
# INDIVIDUAL MODALITY RESULT
# ============================================================================

class ModalityResult(BaseModel):
    """
    Result produced by one individual modality branch.

    Example:

        {
            "modality": "genetic",
            "probability": 0.519817,
            "prediction": "ASD"
        }
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    modality: ModalityName

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="ASD probability produced by the modality model.",
    )

    prediction: PredictionLabel = Field(
        ...,
        description="Binary classification using the 0.5 threshold.",
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold.",
    )

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Probability must be within [0, 1]."
            )

        return float(value)


# ============================================================================
# FUSION CONTRIBUTION
# ============================================================================

class FusionContribution(BaseModel):
    """
    Contribution of one modality to knowledge-level fusion.

    Example:

        Genetic probability  : 0.519817
        Genetic weight       : 0.319370
        Contribution         : 0.166014
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    modality: ModalityName

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    contribution: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


# ============================================================================
# FUSION RESULT
# ============================================================================

class FusionResult(BaseModel):
    """
    Knowledge-level multimodal fusion result.

    This represents the final probability generated from the
    selected modality configuration.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    mode: InferenceMode

    modalities_used: List[ModalityName] = Field(
        ...,
        min_length=1,
        max_length=3,
    )

    final_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final ASD probability.",
    )

    final_prediction: PredictionLabel = Field(
        ...,
        description="Final binary prediction.",
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    fusion_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Verified modality weights used by the fusion pipeline.",
    )

    contributions: List[FusionContribution] = Field(
        default_factory=list,
        description="Per-modality fusion contributions.",
    )

    @field_validator("final_probability")
    @classmethod
    def validate_final_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Final probability must be within [0, 1]."
            )

        return float(value)


class CrossModalResult(BaseModel):
    """Final feature-level cross-modal ASD prediction."""

    model_config = ConfigDict(extra="forbid")

    model: str
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: PredictionLabel
    prediction_numeric: int = Field(
        ...,
        ge=0,
        le=1,
        description="Numeric prediction using the configured threshold.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of prediction: P(ASD) for ASD, 1 - P(ASD) for Control."
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    modalities_used: List[ModalityName] = Field(default_factory=list, min_length=3, max_length=3)
    joint_embedding_dimension: int = Field(default=32, ge=1)



# ============================================================================
# COMPLETE PATIENT RESULT
# ============================================================================

class PatientInferenceResult(BaseModel):
    """
    Complete backend response for one patient.

    This is the main result object eventually returned to the frontend.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    patient_id: str = Field(
        ...,
        min_length=1,
    )

    mode: InferenceMode

    modality_results: List[ModalityResult] = Field(
        default_factory=list,
    )

    fusion: Optional[FusionResult] = None

    cross_modal: Optional[CrossModalResult] = None

    status: Literal[
        "success",
        "error",
    ] = "success"

    message: Optional[str] = None

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "patient_id cannot be empty."
            )

        if "/" in value or "\\" in value:
            raise ValueError(
                "patient_id must not contain path separators."
            )

        return value


# ============================================================================
# MULTI-PATIENT RESULT
# ============================================================================

class BatchInferenceResult(BaseModel):
    """
    Result container for P001-P005 or another batch of patients.

    Used later for backend validation and demo testing.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    results: List[PatientInferenceResult] = Field(
        default_factory=list,
    )

    total_patients: int = Field(
        default=0,
        ge=0,
    )

    successful_patients: int = Field(
        default=0,
        ge=0,
    )

    failed_patients: int = Field(
        default=0,
        ge=0,
    )


# ============================================================================
# ERROR RESPONSE
# ============================================================================

class ErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal["error"] = "error"

    message: str

    detail: Optional[str] = None