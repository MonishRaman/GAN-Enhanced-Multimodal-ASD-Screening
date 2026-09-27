"""
ASD Multimodal Project
FastAPI Application Entry Point
--------------------------------

Step 13:
    Connect the frozen modality services to the frozen fusion service.

This module:
    - exposes the FastAPI API
    - loads demo patient inputs
    - runs genetic / behavioral / environmental inference
    - obtains the three 32-D aligned representations
    - performs M3.2 weighted probability evidence fusion
    - returns validated PatientInferenceResult objects

No model is trained or modified here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.patient_schema import (
    DemoPatientRequest,
    MultimodalInferenceRequest,
)
from backend.schemas.result_schema import (
    BatchInferenceResult,
    ErrorResponse,
    FusionContribution,
    FusionResult,
    CrossModalResult,
    ModalityResult,
    PatientInferenceResult,
)
from backend.services.behavioral_service import BehavioralService
from backend.services.environmental_service import EnvironmentalService
from backend.services.fusion_service import FusionService
from backend.services.cross_modal_service import cross_modal_service
from backend.services.genetic_service import GeneticService
from backend.utils.paths import (
    get_behavioral_input_path,
    get_demo_patient_dir,
    get_environmental_input_path,
    get_genetic_input_path,
    validate_frozen_artifacts,
    validate_project_structure,
)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="ASD Multimodal Analysis API",
    description=(
        "ASD multimodal inference using frozen Genetic, Behavioral, "
        "Environmental, and M3.2 knowledge-level fusion models."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SERVICE INITIALIZATION
# ============================================================================

genetic_service = GeneticService()
behavioral_service = BehavioralService()
environmental_service = EnvironmentalService()
fusion_service = FusionService()


# ============================================================================
# CONSTANTS
# ============================================================================

DECISION_THRESHOLD = 0.5

VALID_MODES = {
    "genetic",
    "behavioral",
    "environmental",
    "genetic_behavioral",
    "genetic_environmental",
    "behavioral_environmental",
    "all_three",
}


# ============================================================================
# HEALTH / STATUS
# ============================================================================

@app.get("/")
def root() -> Dict[str, Any]:
    """
    Basic API information.
    """

    return {
        "status": "success",
        "service": "ASD Multimodal Analysis API",
        "version": "1.0.0",
        "message": "Backend API is running.",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Backend health check.
    """

    return {
        "status": "success",
        "backend": "healthy",
        "services": {
            "genetic": genetic_service is not None,
            "behavioral": behavioral_service is not None,
            "environmental": environmental_service is not None,
            "fusion": fusion_service is not None,
        },
    }


@app.get("/system/status")
def system_status() -> Dict[str, Any]:
    """
    Report project structure and frozen artifact availability.
    """

    return {
        "status": "success",
        "project_structure": validate_project_structure(),
        "frozen_artifacts": validate_frozen_artifacts(),
    }


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _prediction_label(probability: float) -> str:
    """
    Convert probability into the project's standard prediction label.
    """

    return (
        "ASD"
        if float(probability) >= DECISION_THRESHOLD
        else "NON-ASD"
    )


def _safe_probability(value: Any) -> float:
    """
    Convert a model probability into a validated Python float.
    """

    probability = float(value)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Invalid probability returned by model: {probability}"
        )

    return probability


def _extract_single_embedding(
    value: Any,
    modality: str,
) -> np.ndarray:
    """
    Convert a modality aligned representation into one 32-D NumPy vector.
    """

    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError(
            f"{modality} aligned representation has invalid "
            f"shape: {array.shape}"
        )

    if array.shape[0] < 1:
        raise ValueError(
            f"{modality} aligned representation contains no rows."
        )

    if array.shape[1] != 32:
        raise ValueError(
            f"{modality} aligned representation must be 32-D; "
            f"received {array.shape[1]} dimensions."
        )

    return array[0]


def _behavioral_aligned_embedding(
    input_data: pd.DataFrame,
) -> np.ndarray:
    """
    Read the aligned representation produced by BehavioralService.

    The service owns the frozen preprocessing, encoder, alignment,
    scaler, and expert pipeline; this helper only validates its
    returned representation for cross-modal inference.
    """

    result = behavioral_service.predict(input_data)

    if "aligned_embedding_32" not in result:
        raise ValueError(
            "Behavioral service did not return an aligned 32-D representation."
        )

    return _extract_single_embedding(
        result["aligned_embedding_32"],
        "behavioral",
    )


def _run_genetic(
    csv_path: Path,
) -> Dict[str, Any]:
    """
    Run the frozen Genetic pipeline.
    """

    dataframe = pd.read_csv(
        csv_path
    )

    result = genetic_service.process(
        dataframe
    )

    embedding = _extract_single_embedding(
        result["aligned_embedding_32"],
        "genetic",
    )

    probability = result.get(
        "probability"
    )

    if probability is None:
        probability = None

    return {
        "result": result,
        "embedding": embedding,
        "probability": probability,
    }


def _run_behavioral(
    csv_path: Path,
) -> Dict[str, Any]:
    """
    Run the frozen Behavioral pipeline.
    """

    dataframe = pd.read_csv(
        csv_path
    )

    result = behavioral_service.predict(
        dataframe
    )

    embedding = _extract_single_embedding(
        result["aligned_embedding_32"],
        "behavioral",
    )

    probability = _safe_probability(
        result["probability"]
    )

    return {
        "result": result,
        "embedding": embedding,
        "probability": probability,
    }


def _run_environmental(
    csv_path: Path,
) -> Dict[str, Any]:
    """
    Run the frozen Environmental pipeline.
    """

    result = environmental_service.process(
        csv_path
    )

    embedding = _extract_single_embedding(
        result["aligned_embedding_32"],
        "environmental",
    )

    probability = result.get(
        "probability"
    )

    if probability is not None:
        probability = _safe_probability(
            probability
        )

    return {
        "result": result,
        "embedding": embedding,
        "probability": probability,
    }


def _modality_result(
    modality: str,
    probability: float,
) -> ModalityResult:
    """
    Build the standard modality result.
    """

    probability = _safe_probability(
        probability
    )

    return ModalityResult(
        modality=modality,
        probability=probability,
        prediction=_prediction_label(
            probability
        ),
        threshold=DECISION_THRESHOLD,
    )


def _build_fusion_result(
    mode: str,
    modalities: List[str],
    fusion_output: Dict[str, Any],
) -> FusionResult:
    """
    Convert FusionService output into the project's response schema.
    """

    final_probability = _safe_probability(
        fusion_output["probability"]
    )

    contributions: List[FusionContribution] = []

    fusion_weights: Dict[str, float] = {}

    modality_outputs = fusion_output.get(
        "modalities",
        {},
    )

    for modality in modalities:
        item = modality_outputs.get(
            modality
        )

        if item is None:
            continue

        probability = _safe_probability(
            item["probability"]
        )

        weight = float(
            item["weight"]
        )

        contribution = probability * weight

        contributions.append(
            FusionContribution(
                modality=modality,
                probability=probability,
                weight=weight,
                contribution=contribution,
            )
        )

        fusion_weights[modality] = weight

    return FusionResult(
        mode=mode,
        modalities_used=modalities,
        final_probability=final_probability,
        final_prediction=_prediction_label(
            final_probability
        ),
        threshold=float(
            fusion_output.get(
                "decision_threshold",
                DECISION_THRESHOLD,
            )
        ),
        fusion_weights=fusion_weights,
        contributions=contributions,
    )


def _run_demo_patient(
    patient_id: str,
    mode: str,
) -> PatientInferenceResult:
    """
    Run a verified demo patient.

    Demo files are already present under:
        data/demo/P001 ... P005
    """

    patient_id = patient_id.upper()

    patient_dir = get_demo_patient_dir(
        patient_id
    )

    if not patient_dir.exists():
        raise FileNotFoundError(
            f"Demo patient directory not found: {patient_dir}"
        )

    genetic_path = get_genetic_input_path(
        patient_id
    )

    behavioral_path = get_behavioral_input_path(
        patient_id
    )

    environmental_path = get_environmental_input_path(
        patient_id
    )

    requested = {
        "genetic": mode in {
            "genetic",
            "genetic_behavioral",
            "genetic_environmental",
            "all_three",
        },
        "behavioral": mode in {
            "behavioral",
            "genetic_behavioral",
            "behavioral_environmental",
            "all_three",
        },
        "environmental": mode in {
            "environmental",
            "genetic_environmental",
            "behavioral_environmental",
            "all_three",
        },
    }

    for modality, enabled in requested.items():
        if not enabled:
            continue

        path = {
            "genetic": genetic_path,
            "behavioral": behavioral_path,
            "environmental": environmental_path,
        }[modality]

        if not path.exists():
            raise FileNotFoundError(
                f"{modality} input file not found: {path}"
            )

    modality_results: List[ModalityResult] = []

    genetic_output = None
    behavioral_output = None
    environmental_output = None

    if requested["genetic"]:
        genetic_output = _run_genetic(
            genetic_path
        )

        if genetic_output["probability"] is not None:
            modality_results.append(
                _modality_result(
                    "genetic",
                    genetic_output["probability"],
                )
            )

    if requested["behavioral"]:
        behavioral_output = _run_behavioral(
            behavioral_path
        )

        modality_results.append(
            _modality_result(
                "behavioral",
                behavioral_output["probability"],
            )
        )

    if requested["environmental"]:
        environmental_output = _run_environmental(
            environmental_path
        )

        if environmental_output["probability"] is not None:
            modality_results.append(
                _modality_result(
                    "environmental",
                    environmental_output["probability"],
                )
            )

    # ------------------------------------------------------------------------
    # Fusion
    #
    # The current M3.2 FusionService is a three-modality weighted evidence
    # fusion service. Therefore, fusion is performed only when all three
    # representations are available.
    # ------------------------------------------------------------------------

    fusion_result = None
    cross_modal_result = None

    if mode == "all_three":
        if (
            genetic_output is None
            or behavioral_output is None
            or environmental_output is None
        ):
            raise RuntimeError(
                "All-three inference requires all three modality outputs."
            )

        cross_modal_output = cross_modal_service.predict(
            genetic_output["embedding"],
            behavioral_output["embedding"],
            environmental_output["embedding"],
        )

        individual_output = fusion_service.predict(
            genetic_output["embedding"],
            behavioral_output["embedding"],
            environmental_output["embedding"],
        )
        modality_results = [
            _modality_result(
                modality,
                individual_output["modalities"][modality]["probability"],
            )
            for modality in ["genetic", "behavioral", "environmental"]
        ]

        cross_modal_result = CrossModalResult(**cross_modal_output)

    elif mode in {
        "genetic_behavioral",
        "genetic_environmental",
        "behavioral_environmental",
    }:
        # --------------------------------------------------------------------
        # The frozen M3.2 implementation uses all three modality weights.
        # Two-modality execution is therefore reported from the available
        # modality probabilities rather than silently renormalizing weights.
        # --------------------------------------------------------------------

        available_probabilities: Dict[str, float] = {}

        if genetic_output is not None:
            available_probabilities["genetic"] = (
                genetic_output["probability"]
            )

        if behavioral_output is not None:
            available_probabilities["behavioral"] = (
                behavioral_output["probability"]
            )

        if environmental_output is not None:
            available_probabilities["environmental"] = (
                environmental_output["probability"]
            )

        if len(available_probabilities) != 2:
            raise RuntimeError(
                "Two-modality inference requires exactly two modality outputs."
            )

        # Use the verified M3.2 weights and renormalize only across the
        # modalities actually requested for this flexible inference mode.
        configured_weights = {
            modality: fusion_service.__dict__.get(
                f"{modality}_weight",
                None,
            )
            for modality in available_probabilities
        }

        # The public FusionService output exposes the canonical weights only
        # after prediction. Retrieve them from a deterministic all-three
        # prediction using the actual representations is unnecessary here.
        #
        # Instead use the canonical M3.2 values defined in the fusion config.
        canonical_weights = {
            "genetic": 0.3194,
            "behavioral": 0.4002,
            "environmental": 0.2804,
        }

        weight_sum = sum(
            canonical_weights[m]
            for m in available_probabilities
        )

        final_probability = sum(
            (
                canonical_weights[m] / weight_sum
            )
            * available_probabilities[m]
            for m in available_probabilities
        )

        fusion_weights = {
            m: canonical_weights[m] / weight_sum
            for m in available_probabilities
        }

        contributions = [
            FusionContribution(
                modality=m,
                probability=available_probabilities[m],
                weight=fusion_weights[m],
                contribution=(
                    available_probabilities[m]
                    * fusion_weights[m]
                ),
            )
            for m in available_probabilities
        ]

        fusion_result = FusionResult(
            mode=mode,
            modalities_used=list(
                available_probabilities.keys()
            ),
            final_probability=float(
                final_probability
            ),
            final_prediction=_prediction_label(
                final_probability
            ),
            threshold=DECISION_THRESHOLD,
            fusion_weights=fusion_weights,
            contributions=contributions,
        )

    return PatientInferenceResult(
        patient_id=patient_id,
        mode=mode,
        modality_results=modality_results,
        fusion=fusion_result,
        cross_modal=cross_modal_result,
        status="success",
        message="Inference completed successfully.",
    )


# ============================================================================
# DEMO PATIENT ENDPOINT
# ============================================================================

@app.post(
    "/api/demo",
    response_model=PatientInferenceResult,
    responses={
        400: {
            "model": ErrorResponse,
        },
        500: {
            "model": ErrorResponse,
        },
    },
)
def demo_patient(
    request: DemoPatientRequest,
) -> PatientInferenceResult:
    """
    Run one of the verified demo patients P001-P005.
    """

    try:
        return _run_demo_patient(
            request.patient_id,
            request.mode,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc


# ============================================================================
# P001-P005 BATCH ENDPOINT
# ============================================================================

@app.post(
    "/api/demo/batch",
    response_model=BatchInferenceResult,
)
def demo_batch() -> BatchInferenceResult:
    """
    Run the verified P001-P005 demo patients using all three modalities.
    """

    results: List[PatientInferenceResult] = []

    for patient_id in [
        "P001",
        "P002",
        "P003",
        "P004",
        "P005",
    ]:
        try:
            result = _run_demo_patient(
                patient_id,
                "all_three",
            )

        except Exception as exc:
            result = PatientInferenceResult(
                patient_id=patient_id,
                mode="all_three",
                modality_results=[],
                fusion=None,
                status="error",
                message=str(exc),
            )

        results.append(
            result
        )

    successful = sum(
        1
        for result in results
        if result.status == "success"
    )

    failed = len(results) - successful

    return BatchInferenceResult(
        results=results,
        total_patients=len(results),
        successful_patients=successful,
        failed_patients=failed,
    )


# ============================================================================
# GENERAL INFERENCE ENDPOINT
# ============================================================================

@app.post(
    "/api/infer",
    response_model=PatientInferenceResult,
)
def infer(
    request: MultimodalInferenceRequest,
) -> PatientInferenceResult:
    """
    General inference endpoint.

    The current implementation uses the verified demo-style CSV inputs
    when a file_path is supplied in a modality input.

    This endpoint is the backend contract that the frontend will use.
    """

    try:
        if request.mode not in VALID_MODES:
            raise ValueError(
                f"Unsupported inference mode: {request.mode}"
            )

        # --------------------------------------------------------------------
        # Current application-layer implementation:
        #
        # Each supplied modality must provide file_path.
        # The frozen services continue to own preprocessing/inference.
        # --------------------------------------------------------------------

        modality_paths: Dict[str, Path] = {}

        if request.genetic is not None:
            if not request.genetic.file_path:
                raise ValueError(
                    "Genetic inference currently requires file_path."
                )

            modality_paths["genetic"] = Path(
                request.genetic.file_path
            )

        if request.behavioral is not None:
            if not request.behavioral.file_path:
                raise ValueError(
                    "Behavioral inference currently requires file_path."
                )

            modality_paths["behavioral"] = Path(
                request.behavioral.file_path
            )

        if request.environmental is not None:
            if not request.environmental.file_path:
                raise ValueError(
                    "Environmental inference currently requires file_path."
                )

            modality_paths["environmental"] = Path(
                request.environmental.file_path
            )

        # --------------------------------------------------------------------
        # For the first frozen backend phase, require the selected mode to
        # correspond to the supplied modality files.
        # --------------------------------------------------------------------

        expected_modalities = {
            "genetic": ["genetic"],
            "behavioral": ["behavioral"],
            "environmental": ["environmental"],
            "genetic_behavioral": [
                "genetic",
                "behavioral",
            ],
            "genetic_environmental": [
                "genetic",
                "environmental",
            ],
            "behavioral_environmental": [
                "behavioral",
                "environmental",
            ],
            "all_three": [
                "genetic",
                "behavioral",
                "environmental",
            ],
        }[request.mode]

        if set(modality_paths.keys()) != set(
            expected_modalities
        ):
            raise ValueError(
                "Supplied modalities do not match requested mode. "
                f"Expected {expected_modalities}, "
                f"received {list(modality_paths.keys())}."
            )

        # --------------------------------------------------------------------
        # Execute selected modality pipelines.
        # --------------------------------------------------------------------

        modality_results: List[ModalityResult] = []

        genetic_output = None
        behavioral_output = None
        environmental_output = None

        if "genetic" in modality_paths:
            genetic_output = _run_genetic(
                modality_paths["genetic"]
            )

            if genetic_output["probability"] is not None:
                modality_results.append(
                    _modality_result(
                        "genetic",
                        genetic_output["probability"],
                    )
                )

        if "behavioral" in modality_paths:
            behavioral_output = _run_behavioral(
                modality_paths["behavioral"]
            )

            modality_results.append(
                _modality_result(
                    "behavioral",
                    behavioral_output["probability"],
                )
            )

        if "environmental" in modality_paths:
            environmental_output = _run_environmental(
                modality_paths["environmental"]
            )

            if environmental_output["probability"] is not None:
                modality_results.append(
                    _modality_result(
                        "environmental",
                        environmental_output["probability"],
                    )
                )

        # --------------------------------------------------------------------
        # All-three fusion.
        # --------------------------------------------------------------------

        fusion_result = None
        cross_modal_result = None

        if request.mode == "all_three":
            if (
                genetic_output is None
                or behavioral_output is None
                or environmental_output is None
            ):
                raise RuntimeError(
                    "All-three inference requires all three modalities."
                )

            cross_modal_output = cross_modal_service.predict(
                genetic_output["embedding"],
                behavioral_output["embedding"],
                environmental_output["embedding"],
            )

            cross_modal_result = CrossModalResult(**cross_modal_output)

        return PatientInferenceResult(
            patient_id=request.patient_id,
            mode=request.mode,
            modality_results=modality_results,
            fusion=fusion_result,
            cross_modal=cross_modal_result,
            status="success",
            message="Inference completed successfully.",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

# ============================================================================
# FINAL CROSS-MODAL CSV UPLOAD ENDPOINT
# ============================================================================

@app.post("/api/cross-modal/upload", response_model=PatientInferenceResult)
async def cross_modal_upload(
    patient_id: Optional[str] = Form(None),
    genetic_file: Optional[UploadFile] = File(None),
    behavioral_file: Optional[UploadFile] = File(None),
    environmental_file: Optional[UploadFile] = File(None),
) -> PatientInferenceResult:
    """
    Run the final feature-level cross-modal ASD model from three patient CSVs.

    Each CSV must contain exactly one patient row. The existing frozen
    modality services own all preprocessing/encoding/alignment.
    """
    import tempfile

    patient_id = (patient_id or "uploaded-patient").strip()
    if "/" in patient_id or "\\" in patient_id:
        raise HTTPException(status_code=400, detail="Patient ID must not contain path separators.")

    files = {
        "genetic": genetic_file,
        "behavioral": behavioral_file,
        "environmental": environmental_file,
    }

    missing = [name for name, upload in files.items() if upload is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail="Missing required CSV file(s): " + ", ".join(missing),
        )

    for name, upload in files.items():
        filename = (upload.filename or "").lower()
        if not filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"{name.capitalize()} input must be a CSV file.")

    try:
        with tempfile.TemporaryDirectory(prefix="asd_cross_modal_") as temp_dir:
            paths = {}
            for name, upload in files.items():
                path = Path(temp_dir) / f"{name}.csv"
                content = await upload.read()
                if not content:
                    raise ValueError(f"{name.capitalize()} CSV is empty.")
                path.write_bytes(content)
                paths[name] = path

            genetic_df = pd.read_csv(paths["genetic"])
            behavioral_df = pd.read_csv(paths["behavioral"])
            environmental_df = pd.read_csv(paths["environmental"])

            for name, df in {"genetic": genetic_df, "behavioral": behavioral_df, "environmental": environmental_df}.items():
                if df.empty:
                    raise ValueError(f"{name.capitalize()} CSV contains no rows.")
                if len(df) != 1:
                    raise ValueError(
                        f"{name.capitalize()} CSV must contain exactly one patient row; received {len(df)} rows."
                    )

            genetic_output = _run_genetic(paths["genetic"])
            behavioral_output = _run_behavioral(paths["behavioral"])
            environmental_output = _run_environmental(paths["environmental"])

            cross_output = cross_modal_service.predict(
                genetic_output["embedding"],
                behavioral_output["embedding"],
                environmental_output["embedding"],
            )

            cross_result = CrossModalResult(**cross_output)

            # Use the frozen modality experts only for individual reporting.
            # The cross-modal result above remains the sole final prediction.
            individual_output = fusion_service.predict(
                genetic_output["embedding"],
                behavioral_output["embedding"],
                environmental_output["embedding"],
            )
            modality_results = [
                _modality_result(
                    modality,
                    individual_output["modalities"][modality]["probability"],
                )
                for modality in ["genetic", "behavioral", "environmental"]
            ]

            return PatientInferenceResult(
                patient_id=patient_id,
                mode="all_three",
                modality_results=modality_results,
                fusion=None,
                cross_modal=cross_result,
                status="success",
                message="Final cross-modal ASD inference completed successfully.",
            )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cross-modal inference failed: {exc}") from exc
