"""
ASD Multimodal Project
Patient Input Schemas
---------------------

Defines the input contract used by the backend API.

The schema layer is responsible only for:
    - identifying the patient
    - identifying which modalities are available
    - receiving modality input data
    - validating the requested inference mode

It does NOT:
    - load models
    - preprocess model data
    - perform inference
    - calculate fusion probabilities
    - modify frozen model artifacts
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


# ============================================================================
# MODALITY DEFINITIONS
# ============================================================================

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
# PATIENT REQUEST
# ============================================================================

class PatientInferenceRequest(BaseModel):
    """
    Main request received by the inference API.

    Example:

        {
            "patient_id": "P001",
            "modalities": ["genetic", "behavioral", "environmental"],
            "mode": "all_three"
        }

    The actual modality values are supplied separately through
    CSV/file inputs or the corresponding service layer.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    patient_id: str = Field(
        ...,
        min_length=1,
        description="Unique patient identifier.",
    )

    modalities: List[ModalityName] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="Modalities available for this inference request.",
    )

    mode: InferenceMode = Field(
        ...,
        description="Requested one-, two-, or three-modality inference mode.",
    )

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str) -> str:
        """
        Keep patient identifiers safe for local file/path handling.

        Expected examples:
            P001
            P002
            PATIENT_001
        """

        value = value.strip()

        if not value:
            raise ValueError("patient_id cannot be empty.")

        if "/" in value or "\\" in value:
            raise ValueError(
                "patient_id must not contain path separators."
            )

        return value

    @field_validator("modalities")
    @classmethod
    def validate_modalities(
        cls,
        value: List[ModalityName],
    ) -> List[ModalityName]:
        """
        Ensure each modality occurs only once.
        """

        if len(set(value)) != len(value):
            raise ValueError(
                "Each modality may appear only once."
            )

        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(
        cls,
        value: InferenceMode,
    ) -> InferenceMode:
        return value


# ============================================================================
# MODALITY DATA
# ============================================================================

class ModalityInput(BaseModel):
    """
    Generic representation of one modality's input.

    The backend services will interpret the actual data according
    to the frozen model's original preprocessing/inference pipeline.

    This class deliberately does not assume that all three modalities
    have the same feature structure.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    modality: ModalityName

    data: Optional[Dict[str, Any]] = None

    file_path: Optional[str] = None


# ============================================================================
# COMPLETE INFERENCE PAYLOAD
# ============================================================================

class MultimodalInferenceRequest(BaseModel):
    """
    Complete inference payload.

    This is the structure that the FastAPI endpoint can eventually
    receive from the frontend.

    Example:

        {
            "patient_id": "P001",
            "mode": "all_three",
            "genetic": {...},
            "behavioral": {...},
            "environmental": {...}
        }

    Individual modality fields remain optional because the system
    supports ANY-ONE, ANY-TWO, and ALL-THREE inference.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    patient_id: str = Field(
        ...,
        min_length=1,
        description="Unique patient identifier.",
    )

    mode: InferenceMode = Field(
        ...,
        description="Requested inference configuration.",
    )

    genetic: Optional[ModalityInput] = None

    behavioral: Optional[ModalityInput] = None

    environmental: Optional[ModalityInput] = None

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("patient_id cannot be empty.")

        if "/" in value or "\\" in value:
            raise ValueError(
                "patient_id must not contain path separators."
            )

        return value


# ============================================================================
# DEMO PATIENT REQUEST
# ============================================================================

class DemoPatientRequest(BaseModel):
    """
    Request for running one of the verified demo patients.

    Valid demo patients currently available:

        P001
        P002
        P003
        P004
        P005
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    patient_id: str = Field(
        ...,
        description="Demo patient identifier.",
    )

    mode: InferenceMode = Field(
        default="all_three",
        description="Inference configuration for the demo patient.",
    )

    @field_validator("patient_id")
    @classmethod
    def validate_demo_patient_id(cls, value: str) -> str:
        value = value.strip().upper()

        allowed = {
            "P001",
            "P002",
            "P003",
            "P004",
            "P005",
        }

        if value not in allowed:
            raise ValueError(
                "Demo patient must be one of P001, P002, P003, P004, or P005."
            )

        return value


# ============================================================================
# UTILITY
# ============================================================================

def get_requested_modalities(
    request: MultimodalInferenceRequest,
) -> List[str]:
    """
    Return the modalities supplied in the request.

    This helper does not perform inference.
    """

    modalities: List[str] = []

    if request.genetic is not None:
        modalities.append("genetic")

    if request.behavioral is not None:
        modalities.append("behavioral")

    if request.environmental is not None:
        modalities.append("environmental")

    return modalities