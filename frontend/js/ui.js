/* =========================================================
   ASD MULTIMODAL ANALYSIS
   Step 18 — UI Rendering & Interaction Layer
   ========================================================= */

/**
 * Small DOM helper.
 */
function $(id) {
    return document.getElementById(id);
}

function firstElement(ids = []) {
    for (const id of ids) {
        const element = $(id);

        if (element) {
            return element;
        }
    }

    return null;
}

/**
 * Set text safely.
 */
function setText(id, value) {
    const element = $(id);
    if (element) {
        element.textContent = value ?? "—";
    }
}

/**
 * Show / hide an element.
 */
function setVisible(id, visible) {
    const element = $(id);

    if (!element) {
        return;
    }

    element.hidden = !visible;
    element.style.display = visible ? "" : "none";
}


/* =========================================================
   BACKEND STATUS
   ========================================================= */

function setBackendStatus(connected, message = null) {
    const textElement =
        firstElement([
            "connectionText",
            "backendStatus",
            "backend-status"
        ]);

    const indicator = firstElement([
        "connectionIndicator"
    ]);

    if (!textElement && !indicator) {
        return;
    }

    if (connected) {
        if (textElement) {
            textElement.textContent = message || "Backend connected";
            textElement.classList.remove("status-error");
            textElement.classList.add("status-success");
        }

        if (indicator) {
            indicator.classList.remove("offline");
            indicator.classList.add("online");
        }
    } else {
        if (textElement) {
            textElement.textContent = message || "Backend not connected";
            textElement.classList.remove("status-success");
            textElement.classList.add("status-error");
        }

        if (indicator) {
            indicator.classList.remove("online");
            indicator.classList.add("offline");
        }
    }
}


/* =========================================================
   LOADING STATE
   ========================================================= */

function showLoading(show = true) {
    const loading =
        firstElement([
            "loadingSection",
            "loading-state",
            "loadingState"
        ]) ||
        document.querySelector("[data-loading]");

    if (loading) {
        loading.hidden = !show;
        loading.style.display = show ? "" : "none";
    }

    const analyzeButton =
        firstElement([
            "analyzeButton",
            "analyze-btn",
            "analyzeBtn",
            "analyze-patient"
        ]);

    if (analyzeButton) {
        analyzeButton.disabled = show;

        if (show) {
            analyzeButton.dataset.originalText =
                analyzeButton.textContent;

            analyzeButton.textContent = "Analyzing...";
        } else {
            analyzeButton.textContent =
                analyzeButton.dataset.originalText || "Analyze Patient";
        }
    }
}


/* =========================================================
   ERROR MESSAGE
   ========================================================= */

function showError(message) {
    const error =
        firstElement([
            "messageArea",
            "error-message",
            "errorMessage"
        ]) ||
        document.querySelector("[data-error]");

    if (!error) {
        console.error(message);
        return;
    }

    error.textContent = message;
    error.hidden = false;
    error.style.display = "";
    error.classList.add("error");
}


/**
 * Clear any previous error.
 */
function clearError() {
    const error =
        firstElement([
            "messageArea",
            "error-message",
            "errorMessage"
        ]) ||
        document.querySelector("[data-error]");

    if (!error) {
        return;
    }

    error.textContent = "";
    error.hidden = true;
    error.style.display = "none";
    error.classList.remove("error");
}


/* =========================================================
   RESULTS VISIBILITY
   ========================================================= */

function showResults(show = true) {
    const results =
        $("resultsSection") ||
        $("results-section") ||
        $("resultsSection") ||
        $("analysis-results") ||
        document.querySelector("[data-results]");

    if (!results) {
        return;
    }

    results.hidden = !show;
    results.style.display = show ? "" : "none";
}


/* =========================================================
   MODALITY RESULT CARD
   ========================================================= */

function createModalityRow(result) {
    const row = document.createElement("div");
    row.className = "modality-result-row";

    const modality = document.createElement("div");
    modality.className = "modality-name";
    modality.textContent =
        result.modality
            ? String(result.modality).charAt(0).toUpperCase() +
              String(result.modality).slice(1)
            : "Unknown";

    const prediction = document.createElement("div");
    prediction.className = "modality-prediction";
    prediction.textContent = result.prediction || "—";

    const probability = document.createElement("div");
    probability.className = "modality-probability";

    const probabilityValue =
        Number(result.probability);

    probability.textContent =
        Number.isFinite(probabilityValue)
            ? `${(probabilityValue * 100).toFixed(2)}%`
            : "—";

    row.appendChild(modality);
    row.appendChild(prediction);
    row.appendChild(probability);

    return row;
}


/**
 * Render individual modality predictions.
 */
function renderModalityResults(results) {
    const container =
        $("modalityResults") ||
        $("modality-results") ||
        document.querySelector("[data-modality-results]");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!Array.isArray(results) || results.length === 0) {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = "No individual modality results returned.";
        container.appendChild(empty);
        return;
    }

    results.forEach(result => {
        container.appendChild(createModalityRow(result));
    });
}


/* =========================================================
   FUSION WEIGHTS
   ========================================================= */

function renderFusionWeights(weights) {
    const container =
        $("fusion-weights") ||
        $("fusionWeights") ||
        document.querySelector("[data-fusion-weights]");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!weights || typeof weights !== "object") {
        container.textContent = "No fusion weights available.";
        return;
    }

    Object.entries(weights).forEach(([modality, weight]) => {
        const row = document.createElement("div");
        row.className = "fusion-row";

        const name = document.createElement("span");
        name.textContent =
            modality.charAt(0).toUpperCase() +
            modality.slice(1);

        const value = document.createElement("strong");

        const numericWeight = Number(weight);

        value.textContent =
            Number.isFinite(numericWeight)
                ? numericWeight.toFixed(4)
                : "—";

        row.appendChild(name);
        row.appendChild(value);

        container.appendChild(row);
    });
}


/* =========================================================
   FUSION CONTRIBUTIONS
   ========================================================= */

function renderContributions(contributions) {
    const container =
        $("fusionContributions") ||
        $("fusion-contributions") ||
        $("contributions") ||
        document.querySelector("[data-contributions]");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!Array.isArray(contributions) || contributions.length === 0) {
        container.textContent = "No modality contributions available.";
        return;
    }

    contributions.forEach(item => {
        const row = document.createElement("div");
        row.className = "fusion-row";

        const name = document.createElement("span");

        name.textContent =
            item.modality
                ? item.modality.charAt(0).toUpperCase() +
                  item.modality.slice(1)
                : "Unknown";

        const value = document.createElement("strong");

        const contribution =
            Number(item.contribution);

        value.textContent =
            Number.isFinite(contribution)
                ? contribution.toFixed(4)
                : "—";

        row.appendChild(name);
        row.appendChild(value);

        container.appendChild(row);
    });
}


/* =========================================================
   COMPLETE ANALYSIS RESULT
   ========================================================= */

function renderAnalysisResult(data) {
    if (!data) {
        return;
    }

    clearError();
    showResults(true);

    const fusion = data.fusion || {};

    /* Patient */
    setText(
        "resultPatient",
        `Patient: ${data.patient_id || "—"}`
    );

    /* Final prediction */
    setText(
        "finalPrediction",
        fusion.final_prediction || "—"
    );

    setText(
        "resultStatus",
        data.status || "—"
    );

    /* Final probability */
    const probability =
        Number(fusion.final_probability);

    setText(
        "finalProbability",
        Number.isFinite(probability)
            ? `${(probability * 100).toFixed(2)}%`
            : "—"
    );

    const probabilityBar = $("finalProbabilityBar");

    if (probabilityBar) {
        probabilityBar.style.width =
            Number.isFinite(probability)
                ? `${Math.max(0, Math.min(100, probability * 100)).toFixed(2)}%`
                : "0%";
    }

    /* Threshold */
    const threshold =
        Number(fusion.threshold);

    setText(
        "finalThreshold",
        Number.isFinite(threshold)
            ? threshold.toFixed(2)
            : "0.50"
    );

    /* Fusion mode */
    setText(
        "fusionMode",
        fusion.mode || data.mode || "—"
    );

    /* Modalities used */
    setText(
        "modalitiesUsed",
        Array.isArray(fusion.modalities_used)
            ? fusion.modalities_used
                  .map(item =>
                      item.charAt(0).toUpperCase() +
                      item.slice(1)
                  )
                  .join(" • ")
            : "—"
    );

    /* Fusion threshold */
    setText(
        "fusionThreshold",
        Number.isFinite(threshold)
            ? threshold.toFixed(2)
            : "0.50"
    );

    /* Individual modality results */
    const baseModalityResults =
        Array.isArray(data.modality_results)
            ? [...data.modality_results]
            : [];

    const thresholdValue =
        Number.isFinite(threshold)
            ? threshold
            : 0.5;

    if (Array.isArray(fusion.contributions)) {
        const seen = new Set(
            baseModalityResults.map(item => item.modality)
        );

        fusion.contributions.forEach(item => {
            if (!item || !item.modality || seen.has(item.modality)) {
                return;
            }

            const probabilityValue = Number(item.probability);

            if (!Number.isFinite(probabilityValue)) {
                return;
            }

            baseModalityResults.push({
                modality: item.modality,
                probability: probabilityValue,
                prediction:
                    probabilityValue >= thresholdValue
                        ? "ASD"
                        : "NON-ASD"
            });

            seen.add(item.modality);
        });
    }

    renderModalityResults(
        baseModalityResults
    );

    /* Fusion weights */
    renderFusionWeights(
        fusion.fusion_weights || {}
    );

    /* Contributions */
    renderContributions(
        fusion.contributions || []
    );

    const raw = $("rawResponse");

    if (raw) {
        raw.textContent = JSON.stringify(data, null, 2);
    }

    /*
     * Mark the result container with the prediction.
     * This allows CSS to style ASD / NON-ASD differently
     * if those classes exist.
     */
    const resultPrediction =
        $("finalPrediction");

    if (resultPrediction) {
        resultPrediction.classList.remove(
            "prediction-asd",
            "prediction-non-asd"
        );

        if (fusion.final_prediction === "ASD") {
            resultPrediction.classList.add(
                "prediction-asd"
            );
        } else if (
            fusion.final_prediction === "NON-ASD"
        ) {
            resultPrediction.classList.add(
                "prediction-non-asd"
            );
        }
    }
}


/* =========================================================
   CLEAR FORM / RESULTS
   ========================================================= */

function clearAnalysisUI() {
    clearError();
    showLoading(false);
    showResults(false);

    setText("resultPatient", "Patient: —");
    setText("resultStatus", "—");
    setText("finalPrediction", "—");
    setText("finalProbability", "—");
    setText("finalThreshold", "0.50");
    setText("fusionMode", "—");
    setText("modalitiesUsed", "—");
    setText("fusionThreshold", "0.50");

    const probabilityBar = $("finalProbabilityBar");

    if (probabilityBar) {
        probabilityBar.style.width = "0%";
    }

    const modalityContainer =
        $("modality-results") ||
        $("modalityResults") ||
        document.querySelector("[data-modality-results]");

    if (modalityContainer) {
        modalityContainer.innerHTML = "";
    }

    const weightsContainer =
        $("fusion-weights") ||
        $("fusionWeights") ||
        document.querySelector("[data-fusion-weights]");

    if (weightsContainer) {
        weightsContainer.innerHTML = "";
    }

    const contributionsContainer =
        $("fusionContributions") ||
        $("fusion-contributions") ||
        $("contributions") ||
        document.querySelector("[data-contributions]");

    if (contributionsContainer) {
        contributionsContainer.innerHTML = "";
    }

    const raw = $("rawResponse");

    if (raw) {
        raw.textContent = "—";
    }
}


/* =========================================================
   INITIAL UI STATE
   ========================================================= */

function initializeUI() {
    clearError();
    showLoading(false);

    /*
     * Results should not appear as active results when the
     * application first opens.
     */
    const results =
        $("resultsSection") ||
        $("results-section") ||
        $("analysis-results") ||
        document.querySelector("[data-results]");

    if (results) {
        results.hidden = true;
        results.style.display = "none";
    }

    console.log("UI layer initialized.");
}


/* =========================================================
   GLOBAL EXPORTS
   ========================================================= */

window.ASD_UI = {
    setBackendStatus,
    showLoading,
    showError,
    clearError,
    showResults,
    renderModalityResults,
    renderFusionWeights,
    renderContributions,
    renderAnalysisResult,
    clearAnalysisUI,
    initializeUI
};


/* =========================================================
   START
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    initializeUI
);