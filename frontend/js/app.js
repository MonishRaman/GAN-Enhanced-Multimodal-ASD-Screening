/* =========================================================
   ASD MULTIMODAL ANALYSIS
   Application Controller — Hierarchical Attention MINN
   (Objective 3 + 4)
   ========================================================= */

(function () {
    "use strict";

    // ── Config ─────────────────────────────────────────────

    const BACKEND_BASE_URL =
        (window.ASD_API && window.ASD_API.API_BASE_URL) ||
        "http://127.0.0.1:8000";

    const DEMO_PATIENTS = ["P001", "P002", "P003", "P004", "P005"];

    // ── DOM helpers ────────────────────────────────────────

    function findEl(...selectors) {
        for (const s of selectors) {
            const el = document.querySelector(s);
            if (el) return el;
        }
        return null;
    }

    function getPatientInput()   { return findEl("#patient-id","#patientId","[name='patient_id']","input[type='text']"); }
    function getAnalyzeButton()  { return findEl("#analyzeButton","#analyze-btn","[data-action='analyze']","button"); }
    function getClearButton()    { return findEl("#clearButton","#clear-btn","[data-action='clear']"); }

    function getUploadedFiles() {
        return {
            genetic:       findEl("#geneticFile","#genetic-file")?.files?.[0]       || null,
            behavioral:    findEl("#behavioralFile","#behavioral-file")?.files?.[0] || null,
            environmental: findEl("#environmentalFile","#environmental-file")?.files?.[0] || null
        };
    }

    // ── Status ─────────────────────────────────────────────

    function setBackendStatus(connected, message) {
        if (window.ASD_UI && typeof window.ASD_UI.setBackendStatus === "function") {
            window.ASD_UI.setBackendStatus(connected, message);
        }
    }

    // ── Loading ────────────────────────────────────────────

    function setLoading(isLoading) {
        if (window.ASD_UI && typeof window.ASD_UI.showLoading === "function") {
            window.ASD_UI.showLoading(isLoading);
        }
    }

    // ── API Calls ──────────────────────────────────────────

    /**
     * PRIMARY PATH: Demo patient → Hierarchical MINN explain endpoint.
     * GET /api/minn/explain/demo/{patient_id}
     */
    async function callMINNExplain(patientId) {
        if (window.ASD_API && typeof window.ASD_API.analyzeMINNDemo === "function") {
            return window.ASD_API.analyzeMINNDemo(patientId);
        }
        // Fallback inline fetch
        const r = await fetch(`${BACKEND_BASE_URL}/api/minn/explain/demo/${encodeURIComponent(patientId)}`);
        const d = await r.json();
        if (!r.ok) throw new Error(d?.detail || `HTTP ${r.status}`);
        return d;
    }

    /**
     * UPLOAD PATH: 3 CSV files → cross-modal upload endpoint.
     * POST /api/cross-modal/upload
     */
    async function callUploadAnalyze(patientId, files) {
        if (window.ASD_API && typeof window.ASD_API.analyzeCrossModalUpload === "function") {
            return window.ASD_API.analyzeCrossModalUpload(patientId, files);
        }
        throw new Error("Upload API is unavailable.");
    }

    // ── Render ─────────────────────────────────────────────

    function renderResult(result) {
        if (window.ASD_UI && typeof window.ASD_UI.renderAnalysisResult === "function") {
            window.ASD_UI.renderAnalysisResult(result);
            return;
        }
        console.log("Backend result:", result);
    }

    // ── Error ──────────────────────────────────────────────

    function showError(error) {
        const message =
            (window.ASD_API && typeof window.ASD_API.getApiErrorMessage === "function")
                ? window.ASD_API.getApiErrorMessage(error)
                : (error?.message || "An unexpected error occurred.");

        console.error("ASD Analysis Error:", error);

        if (window.ASD_UI && typeof window.ASD_UI.showError === "function") {
            window.ASD_UI.showError(message);
        }
    }

    // ── Validation ─────────────────────────────────────────

    function validatePatientId(patientId) {
        if (!patientId || !patientId.trim()) throw new Error("Please enter a patient ID.");
        return patientId.trim().toUpperCase();
    }

    // ── Analyze handler ────────────────────────────────────

    async function handleAnalyze() {
        try {
            const patientInput   = getPatientInput();
            const uploadedFiles  = getUploadedFiles();
            const fileCount      = Object.values(uploadedFiles).filter(Boolean).length;

            if (fileCount > 0 && fileCount < 3) {
                throw new Error("Please select all three CSV files (Genetic, Behavioral, Environmental).");
            }

            const rawId      = patientInput ? patientInput.value : "";
            const patientId  = fileCount === 3
                ? rawId.trim().toUpperCase()
                : validatePatientId(rawId);

            setLoading(true);
            setBackendStatus(null, "Connecting to backend…");

            const result = fileCount === 3
                ? await callUploadAnalyze(patientId, uploadedFiles)
                : await callMINNExplain(patientId);

            setBackendStatus(true, "Backend connected");
            renderResult(result);

            console.log("MINN Analysis result:", result);

        } catch (error) {
            const isNetworkError = error instanceof TypeError;
            setBackendStatus(
                !isNetworkError,
                isNetworkError ? "Backend connection failed" : "Backend connected"
            );
            showError(error);
        } finally {
            setLoading(false);
        }
    }

    // ── Clear handler ──────────────────────────────────────

    function handleClear() {
        const patientInput = getPatientInput();
        if (patientInput) patientInput.value = "";

        ["#geneticFile", "#behavioralFile", "#environmentalFile"].forEach(sel => {
            const f = document.querySelector(sel);
            if (f) f.value = "";
        });

        if (window.ASD_UI && typeof window.ASD_UI.clearAnalysisUI === "function") {
            window.ASD_UI.clearAnalysisUI();
        }
    }

    // ── Backend health ─────────────────────────────────────

    async function checkBackend() {
        try {
            const healthy =
                (window.ASD_API && typeof window.ASD_API.checkBackendHealth === "function")
                    ? await window.ASD_API.checkBackendHealth()
                    : (await fetch(`${BACKEND_BASE_URL}/health`)).ok;

            setBackendStatus(healthy, healthy ? "Backend connected" : "Backend not connected");
            return healthy;
        } catch (_) {
            setBackendStatus(false, "Backend not connected");
            return false;
        }
    }

    // ── Event binding ──────────────────────────────────────

    function bindEvents() {
        const analyzeBtn = getAnalyzeButton();
        const clearBtn   = getClearButton();
        const patientIn  = getPatientInput();

        if (analyzeBtn) {
            analyzeBtn.addEventListener("click", e => { e.preventDefault(); handleAnalyze(); });
        } else {
            console.warn("Analyze button not found.");
        }

        if (clearBtn) {
            clearBtn.addEventListener("click", e => { e.preventDefault(); handleClear(); });
        }

        if (patientIn) {
            patientIn.addEventListener("keydown", e => {
                if (e.key === "Enter") { e.preventDefault(); handleAnalyze(); }
            });
        }

        // Demo patient quick-select buttons
        document.querySelectorAll("[data-demo-patient]").forEach(btn => {
            btn.addEventListener("click", () => {
                const pid = btn.dataset.demoPatient;
                if (patientIn) patientIn.value = pid;
                handleAnalyze();
            });
        });
    }

    // ── Initialize ─────────────────────────────────────────

    function initializeApp() {
        console.log("================================================");
        console.log("ASD Multimodal — Hierarchical Attention MINN UI");
        console.log("Demo patients:", DEMO_PATIENTS.join(", "));
        console.log("Backend:", BACKEND_BASE_URL);
        console.log("================================================");

        bindEvents();

        if (window.ASD_UI && typeof window.ASD_UI.clearAnalysisUI === "function") {
            window.ASD_UI.clearAnalysisUI();
        }

        checkBackend();
    }

    // ── Public API ─────────────────────────────────────────

    window.ASD_APP = {
        initialize:   initializeApp,
        analyze:      handleAnalyze,
        clear:        handleClear,
        checkBackend: checkBackend
    };

    // ── Boot ───────────────────────────────────────────────

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeApp);
    } else {
        initializeApp();
    }

})();