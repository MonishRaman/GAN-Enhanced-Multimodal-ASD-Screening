"""
ASD Multimodal Analysis Backend
--------------------------------
Centralized project paths.

This module provides local, VS Code-compatible paths to the frozen
ASD multimodal project artifacts.

IMPORTANT:
- No model is trained here.
- No model is modified here.
- No Google Colab /content paths are used.
- All paths are resolved relative to the project root.
"""

from pathlib import Path


# ============================================================================
# PROJECT ROOT
# ============================================================================

# paths.py
#   -> utils/
#       -> backend/
#           -> PROJECT ROOT
#
# Therefore parents[2] points to ASD_Multimodal_UI/

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# MAIN PROJECT DIRECTORIES
# ============================================================================

BACKEND_DIR = PROJECT_ROOT / "backend"

GENETIC_MODEL_DIR = PROJECT_ROOT / "Genetic_Model"

BEHAVIORAL_MODEL_DIR = PROJECT_ROOT / "Behavioral_Model"

ENVIRONMENTAL_MODEL_DIR = PROJECT_ROOT / "Environmental_Model"

FUSION_DIR = PROJECT_ROOT / "05_multimodal_fusion"

DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"

ORIGINAL_DATA_DIR = PROJECT_ROOT / "data" / "original"

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DOCS_DIR = PROJECT_ROOT / "docs"


# ============================================================================
# ENVIRONMENTAL MODEL ARTIFACTS
# ============================================================================

ENVIRONMENTAL_PREPROCESSOR = (
    ENVIRONMENTAL_MODEL_DIR
    / "02_preprocessing"
    / "environmental_preprocessor.joblib"
)

ENVIRONMENTAL_ENCODER = (
    ENVIRONMENTAL_MODEL_DIR
    / "03_representation_learning"
    / "environmental_encoder_128d.pt"
)

ENVIRONMENTAL_ALIGNMENT = (
    ENVIRONMENTAL_MODEL_DIR
    / "04_cross_modal_alignment"
    / "environmental_128_to_32_alignment.pt"
)


# ============================================================================
# MULTIMODAL FUSION ARTIFACTS
# ============================================================================

ENVIRONMENTAL_EXPERT = (
    FUSION_DIR
    / "M3_1_environmental_expert.pt"
)

FUSION_CHECKPOINT = (
    FUSION_DIR
    / "M3_multimodal_knowledge_fusion.pt"
)

FUSION_WEIGHTS_FILE = (
    FUSION_DIR
    / "M3_1_fusion_weights.csv"
)

FUSION_PREDICTIONS_FILE = (
    FUSION_DIR
    / "M3_multimodal_knowledge_fusion_predictions.csv"
)

FUSION_REPORT_FILE = (
    FUSION_DIR
    / "M3_multimodal_knowledge_fusion_report.txt"
)

BALANCED_FUSION_REPORT_FILE = (
    FUSION_DIR
    / "M3_1_balanced_knowledge_fusion_report.txt"
)


# ============================================================================
# FINAL VERIFIED RESULTS
# ============================================================================

FINAL_THREE_MODAL_RESULTS = (
    FUSION_DIR
    / "M4_final_P001_P005_three_modal_fusion_results.csv"
)

FINAL_THREE_MODAL_REPORT = (
    FUSION_DIR
    / "M4_final_three_modal_fusion_validation_report.txt"
)

FINAL_FLEXIBLE_RESULTS = (
    FUSION_DIR
    / "M4_final_P001_P005_flexible_multimodal_results.csv"
)

FINAL_FLEXIBLE_REPORT = (
    FUSION_DIR
    / "M4_final_flexible_multimodal_validation_report.txt"
)


# ============================================================================
# AUDIT / PROJECT FREEZE ARTIFACTS
# ============================================================================

FINAL_AUDIT_DIR = FUSION_DIR / "FINAL_PROJECT_AUDIT"

FINAL_ARTIFACT_MANIFEST = (
    FINAL_AUDIT_DIR
    / "M4_FINAL_ARTIFACT_MANIFEST.json"
)

FINAL_PROJECT_FREEZE_REPORT = (
    FINAL_AUDIT_DIR
    / "M4_FINAL_PROJECT_FREEZE_REPORT.txt"
)


# ============================================================================
# DEMO PATIENT PATHS
# ============================================================================

def get_demo_patient_dir(patient_id: str) -> Path:
    """
    Return the directory containing the three modality inputs
    for a demo patient.

    Example:
        get_demo_patient_dir("P001")

    Returns:
        data/demo/P001/
    """
    return DEMO_DATA_DIR / patient_id


def get_genetic_input_path(patient_id: str) -> Path:
    """Return the Genetic input CSV for a demo patient."""
    return get_demo_patient_dir(patient_id) / "genetic_input.csv"


def get_behavioral_input_path(patient_id: str) -> Path:
    """Return the Behavioral input CSV for a demo patient."""
    return get_demo_patient_dir(patient_id) / "behavioral_input.csv"


def get_environmental_input_path(patient_id: str) -> Path:
    """Return the Environmental input CSV for a demo patient."""
    return get_demo_patient_dir(patient_id) / "environmental_input.csv"


# ============================================================================
# GENERATED RESULT PATHS
# ============================================================================

GENERATED_RESULTS_DIR = RESULTS_DIR / "generated"

DEMO_RESULTS_DIR = RESULTS_DIR / "demo"


def get_generated_result_path(patient_id: str) -> Path:
    """
    Return the default location for a newly generated patient result.
    """
    return GENERATED_RESULTS_DIR / f"{patient_id}_multimodal_result.json"


# ============================================================================
# PATH VALIDATION
# ============================================================================

def validate_project_structure() -> dict:
    """
    Check that the major frozen project directories exist.

    This function does not load or modify any model.
    """

    required_paths = {
        "project_root": PROJECT_ROOT,
        "genetic_model": GENETIC_MODEL_DIR,
        "behavioral_model": BEHAVIORAL_MODEL_DIR,
        "environmental_model": ENVIRONMENTAL_MODEL_DIR,
        "fusion": FUSION_DIR,
        "demo_data": DEMO_DATA_DIR,
    }

    return {
        name: path.exists()
        for name, path in required_paths.items()
    }


# ============================================================================
# FROZEN ARTIFACT VALIDATION
# ============================================================================

def validate_frozen_artifacts() -> dict:
    """
    Check the presence of the critical frozen inference artifacts.

    This performs existence checks only.
    It does not retrain, overwrite, or modify anything.
    """

    required_artifacts = {
        "environmental_preprocessor": ENVIRONMENTAL_PREPROCESSOR,
        "environmental_encoder": ENVIRONMENTAL_ENCODER,
        "environmental_alignment": ENVIRONMENTAL_ALIGNMENT,
        "environmental_expert": ENVIRONMENTAL_EXPERT,
        "fusion_checkpoint": FUSION_CHECKPOINT,
        "fusion_weights": FUSION_WEIGHTS_FILE,
        "final_three_modal_results": FINAL_THREE_MODAL_RESULTS,
        "final_flexible_results": FINAL_FLEXIBLE_RESULTS,
    }

    return {
        name: path.exists()
        for name, path in required_artifacts.items()
    }