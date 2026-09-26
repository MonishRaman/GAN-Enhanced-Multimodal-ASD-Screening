/* =========================================================
   ASD MULTIMODAL ANALYSIS
   Step 19 — Main Frontend Application Controller
   ========================================================= */

(function () {
    "use strict";

    // ---------------------------------------------------------
    // CONFIGURATION
    // ---------------------------------------------------------

    const BACKEND_BASE_URL = "http://127.0.0.1:8001";

    const DEMO_PATIENTS = [
        "P001",
        "P002",
        "P003",
        "P004",
        "P005"
    ];

    // ---------------------------------------------------------
    // DOM HELPERS
    // ---------------------------------------------------------

    function findElement(...selectors) {
        for (const selector of selectors) {
            const element = document.querySelector(selector);

            if (element) {
                return element;
            }
        }

        return null;
    }

    function getPatientInput() {
        return findElement(
            "#patient-id",
            "#patientId",
            "[name='patient_id']",
            "input[type='text']"
        );
    }

    function getModeInput() {
        return findElement(
            "#analysis-mode",
            "#analysisMode",
            "[name='mode']",
            "select"
        );
    }

    function getAnalyzeButton() {
        return findElement(
            "#analyze-btn",
            "#analyzeButton",
            "[data-action='analyze']",
            "button"
        );
    }

    function getClearButton() {
        return findElement(
            "#clear-btn",
            "#clearButton",
            "[data-action='clear']"
        );
    }

    function getUploadedFiles() {
        return {
            genetic: findElement("#geneticFile", "#genetic-file")?.files?.[0] || null,
            behavioral: findElement("#behavioralFile", "#behavioral-file")?.files?.[0] || null,
            environmental: findElement("#environmentalFile", "#environmental-file")?.files?.[0] || null
        };
    }

    // ---------------------------------------------------------
    // STATUS
    // ---------------------------------------------------------

    function setBackendStatus(connected, message) {
        if (
            window.ASD_UI &&
            typeof window.ASD_UI.setBackendStatus === "function"
        ) {
            window.ASD_UI.setBackendStatus(connected, message);
            return;
        }

        const statusElement = findElement(
            "#connectionText",
            "#backendStatus",
            "#backend-status"
        );

        if (!statusElement) {
            return;
        }

        statusElement.textContent = message;

        statusElement.classList.remove(
            "status-success",
            "status-error",
            "status-warning"
        );

        if (connected === true) {
            statusElement.classList.add("status-success");
        } else if (connected === false) {
            statusElement.classList.add("status-error");
        } else {
            statusElement.classList.add("status-warning");
        }
    }

    // ---------------------------------------------------------
    // LOADING STATE
    // ---------------------------------------------------------

    function setLoading(isLoading) {
        if (
            window.ASD_UI &&
            typeof window.ASD_UI.showLoading === "function"
        ) {
            window.ASD_UI.showLoading(isLoading);
            return;
        }

        const analyzeButton = getAnalyzeButton();

        if (analyzeButton) {
            analyzeButton.disabled = isLoading;

            if (isLoading) {
                analyzeButton.dataset.originalText =
                    analyzeButton.textContent;

                analyzeButton.textContent = "Analyzing...";
            } else {
                analyzeButton.textContent =
                    analyzeButton.dataset.originalText ||
                    "Analyze Patient";
            }
        }

        const loadingElement = findElement(
            "#loadingSection",
            "#loading-state",
            ".loading-state"
        );

        if (loadingElement) {
            loadingElement.hidden = !isLoading;
        }
    }

    // ---------------------------------------------------------
    // API REQUEST
    // ---------------------------------------------------------

    async function callDemoAPI(patientId, mode) {

        const response = await fetch(
            `${BACKEND_BASE_URL}/api/demo`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    patient_id: patientId,
                    mode: mode
                })
            }
        );

        let payload = null;

        try {
            payload = await response.json();
        } catch (error) {
            payload = {
                status: "error",
                message: "Backend returned an invalid JSON response."
            };
        }

        if (!response.ok) {
            const message =
                payload?.detail ||
                payload?.message ||
                `Backend request failed with status ${response.status}.`;

            throw new Error(message);
        }

        return payload;
    }

    // ---------------------------------------------------------
    // API LAYER COMPATIBILITY
    // ---------------------------------------------------------

    async function analyzePatient(patientId, mode) {

        /*
         * If api.js exposes a reusable API function,
         * use it. Otherwise use the backend endpoint directly.
         */

        if (
            window.ASD_API &&
            typeof window.ASD_API.analyzePatient === "function"
        ) {
            return window.ASD_API.analyzePatient(
                patientId,
                mode
            );
        }

        if (
            window.ASD_API &&
            typeof window.ASD_API.analyzeDemoPatient === "function"
        ) {
            return window.ASD_API.analyzeDemoPatient(
                patientId,
                mode
            );
        }

        if (
            window.API &&
            typeof window.API.demoPatient === "function"
        ) {
            return window.API.demoPatient(
                patientId,
                mode
            );
        }

        if (
            window.API &&
            typeof window.API.analyzeDemoPatient === "function"
        ) {
            return window.API.analyzeDemoPatient(
                patientId,
                mode
            );
        }

        return callDemoAPI(
            patientId,
            mode
        );
    }

    async function analyzeUploadedPatient(patientId, files) {
        if (
            window.ASD_API &&
            typeof window.ASD_API.analyzeCrossModalUpload === "function"
        ) {
            return window.ASD_API.analyzeCrossModalUpload(patientId, files);
        }

        throw new Error("Cross-modal upload API is not available.");
    }

    // ---------------------------------------------------------
    // UI RENDERING COMPATIBILITY
    // ---------------------------------------------------------

    function renderResult(result) {

        /*
         * ui.js is responsible for presentation.
         * We support several possible exported function names
         * so the controller remains compatible with the UI layer.
         */

        if (
            window.ASD_UI &&
            typeof window.ASD_UI.renderAnalysisResult === "function"
        ) {
            window.ASD_UI.renderAnalysisResult(result);
            return;
        }

        if (
            window.ASD_UI &&
            typeof window.ASD_UI.renderResult === "function"
        ) {
            window.ASD_UI.renderResult(result);
            return;
        }

        if (
            window.UI &&
            typeof window.UI.renderAnalysisResult === "function"
        ) {
            window.UI.renderAnalysisResult(result);
            return;
        }

        if (
            window.UI &&
            typeof window.UI.renderResult === "function"
        ) {
            window.UI.renderResult(result);
            return;
        }

        /*
         * Last-resort rendering.
         * This ensures the application still shows the backend
         * response even if ui.js does not expose a renderer.
         */

        renderFallbackResult(result);
    }

    // ---------------------------------------------------------
    // FALLBACK RESULT RENDERER
    // ---------------------------------------------------------

    function renderFallbackResult(result) {

        const resultsContainer = findElement(
            "#results",
            "#analysis-results",
            ".analysis-results"
        );

        if (!resultsContainer) {
            console.log(
                "Backend result:",
                result
            );

            return;
        }

        const fusion = result?.fusion || {};

        const probability =
            typeof fusion.final_probability === "number"
                ? fusion.final_probability
                : null;

        const prediction =
            fusion.final_prediction ||
            "—";

        const modalities =
            Array.isArray(fusion.modalities_used)
                ? fusion.modalities_used.join(", ")
                : "—";

        resultsContainer.innerHTML = `
            <div class="result-summary">

                <h3>Analysis Complete</h3>

                <p>
                    <strong>Patient:</strong>
                    ${escapeHTML(result?.patient_id || "—")}
                </p>

                <p>
                    <strong>Prediction:</strong>
                    ${escapeHTML(prediction)}
                </p>

                <p>
                    <strong>Final Probability:</strong>
                    ${
                        probability !== null
                            ? `${(probability * 100).toFixed(2)}%`
                            : "—"
                    }
                </p>

                <p>
                    <strong>Modalities Used:</strong>
                    ${escapeHTML(modalities)}
                </p>

            </div>
        `;
    }

    // ---------------------------------------------------------
    // ERROR RENDERING
    // ---------------------------------------------------------

    function showError(error) {

        const message =
            window.ASD_API &&
            typeof window.ASD_API.getApiErrorMessage === "function"
                ? window.ASD_API.getApiErrorMessage(error)
                : (error && error.message)
                    ? error.message
                    : "An unexpected backend error occurred.";

        console.error(
            "ASD Multimodal Analysis Error:",
            error
        );

        if (
            window.ASD_UI &&
            typeof window.ASD_UI.showError === "function"
        ) {
            window.ASD_UI.showError(
                message
            );

            return;
        }

        if (
            window.UI &&
            typeof window.UI.showError === "function"
        ) {
            window.UI.showError(
                message
            );

            return;
        }

        const errorElement = findElement(
            "#error-message",
            "#error",
            ".error-message"
        );

        if (errorElement) {
            errorElement.textContent =
                message;

            errorElement.hidden = false;
        } else {
            alert(
                `Analysis failed:\n\n${error.message}`
            );
        }
    }

    // ---------------------------------------------------------
    // VALIDATION
    // ---------------------------------------------------------

    function validatePatientId(patientId) {

        if (!patientId) {
            throw new Error(
                "Please enter a patient ID."
            );
        }

        const normalized =
            patientId.trim().toUpperCase();

        if (!normalized) {
            throw new Error(
                "Patient ID cannot be empty."
            );
        }

        return normalized;
    }

    function validateMode(mode) {

        const validModes = [
            "genetic",
            "behavioral",
            "environmental",
            "genetic_behavioral",
            "genetic_environmental",
            "behavioral_environmental",
            "all_three"
        ];

        if (!validModes.includes(mode)) {
            throw new Error(
                `Unsupported analysis mode: ${mode}`
            );
        }

        return mode;
    }

    // ---------------------------------------------------------
    // ANALYZE BUTTON HANDLER
    // ---------------------------------------------------------

    async function handleAnalyze() {

        try {

            const patientInput =
                getPatientInput();

            const modeInput =
                getModeInput();

            const mode =
                validateMode(
                    modeInput
                        ? modeInput.value
                        : "all_three"
                );

            const uploadedFiles = getUploadedFiles();
            const selectedFileCount = Object.values(uploadedFiles)
                .filter(Boolean)
                .length;

            if (selectedFileCount > 0 && selectedFileCount < 3) {
                throw new Error(
                    "Please select Genetic, Behavioral, and Environmental CSV files."
                );
            }

            const enteredPatientId = patientInput
                ? patientInput.value
                : "";

            const patientId = selectedFileCount === 3
                ? enteredPatientId.trim().toUpperCase()
                : validatePatientId(enteredPatientId);

            setLoading(true);

            setBackendStatus(
                null,
                "Connecting to backend..."
            );

            const result = selectedFileCount === 3
                ? await analyzeUploadedPatient(patientId, uploadedFiles)
                : await analyzePatient(patientId, mode);

            setBackendStatus(
                true,
                "Backend connected"
            );

            renderResult(result);

            console.log(
                "Analysis result:",
                result
            );

        } catch (error) {

            const isConnectionError =
                error instanceof TypeError;

            setBackendStatus(
                !isConnectionError,
                isConnectionError
                    ? "Backend connection failed"
                    : "Backend connected"
            );

            showError(error);

        } finally {

            setLoading(false);

        }
    }

    // ---------------------------------------------------------
    // CLEAR HANDLER
    // ---------------------------------------------------------

    function handleClear() {

        const patientInput =
            getPatientInput();

        if (patientInput) {
            patientInput.value = "";
        }

        ["#geneticFile", "#behavioralFile", "#environmentalFile"].forEach(
            selector => {
                const fileInput = document.querySelector(selector);
                if (fileInput) {
                    fileInput.value = "";
                }
            }
        );

        const errorElement = findElement(
            "#error-message",
            "#error",
            ".error-message"
        );

        if (errorElement) {
            errorElement.hidden = true;
            errorElement.textContent = "";
        }

        if (
            window.ASD_UI &&
            typeof window.ASD_UI.clearAnalysisUI === "function"
        ) {
            window.ASD_UI.clearAnalysisUI();
            return;
        }

        if (
            window.UI &&
            typeof window.UI.clearResults === "function"
        ) {
            window.UI.clearResults();
            return;
        }

        console.log(
            "Analysis form cleared."
        );
    }

    // ---------------------------------------------------------
    // BACKEND HEALTH CHECK
    // ---------------------------------------------------------

    async function checkBackend() {

        try {

            if (
                window.ASD_API &&
                typeof window.ASD_API.checkBackendHealth === "function"
            ) {
                const healthy = await window.ASD_API.checkBackendHealth();

                setBackendStatus(
                    healthy,
                    healthy
                        ? "Backend connected"
                        : "Backend not connected"
                );

                return healthy;
            }

            const response =
                await fetch(
                    `${BACKEND_BASE_URL}/health`,
                    {
                        method: "GET"
                    }
                );

            if (response.ok) {

                setBackendStatus(
                    true,
                    "Backend connected"
                );

                return true;
            }

        } catch (error) {

            console.warn(
                "Backend health check failed:",
                error
            );

        }

        setBackendStatus(
            false,
            "Backend not connected"
        );

        return false;
    }

    // ---------------------------------------------------------
    // HTML ESCAPING
    // ---------------------------------------------------------

    function escapeHTML(value) {

        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    // ---------------------------------------------------------
    // EVENT BINDING
    // ---------------------------------------------------------

    function bindEvents() {

        const analyzeButton =
            getAnalyzeButton();

        const clearButton =
            getClearButton();

        if (analyzeButton) {

            analyzeButton.addEventListener(
                "click",
                function (event) {

                    event.preventDefault();

                    handleAnalyze();

                }
            );

        } else {

            console.warn(
                "Analyze button not found."
            );

        }

        if (clearButton) {

            clearButton.addEventListener(
                "click",
                function (event) {

                    event.preventDefault();

                    handleClear();

                }
            );

        }

        const patientInput =
            getPatientInput();

        if (patientInput) {

            patientInput.addEventListener(
                "keydown",
                function (event) {

                    if (
                        event.key === "Enter"
                    ) {

                        event.preventDefault();

                        handleAnalyze();

                    }

                }
            );

        }

    }

    // ---------------------------------------------------------
    // APPLICATION INITIALIZATION
    // ---------------------------------------------------------

    function initializeApp() {

        console.log(
            "========================================"
        );

        console.log(
            "ASD MULTIMODAL ANALYSIS"
        );

        console.log(
            "Frontend application initializing..."
        );

        console.log(
            "Demo patients:",
            DEMO_PATIENTS.join(", ")
        );

        console.log(
            "Backend:",
            BACKEND_BASE_URL
        );

        console.log(
            "========================================"
        );

        bindEvents();

        if (
            window.ASD_UI &&
            typeof window.ASD_UI.clearAnalysisUI === "function"
        ) {
            window.ASD_UI.clearAnalysisUI();
        }

        checkBackend();

    }

    // ---------------------------------------------------------
    // PUBLIC APPLICATION API
    // ---------------------------------------------------------

    window.ASD_APP = {

        initialize: initializeApp,

        analyze: handleAnalyze,

        clear: handleClear,

        checkBackend: checkBackend

    };

    // ---------------------------------------------------------
    // START APPLICATION
    // ---------------------------------------------------------

    if (
        document.readyState === "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            initializeApp
        );

    } else {

        initializeApp();

    }

})();