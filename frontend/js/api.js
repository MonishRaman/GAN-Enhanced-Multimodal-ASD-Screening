// ============================================================
// ASD MULTIMODAL ANALYSIS — API Layer
// Routes all calls through the new Hierarchical MINN endpoints
// (Objective 3 + 4).  Legacy endpoints remain for upload flow.
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

function extractErrorMessage(data, status) {
    const detail = data?.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0];
        if (typeof first === "string") return first;
        if (first && typeof first.msg === "string") return first.msg;
    }
    if (detail && typeof detail === "object") {
        if (typeof detail.message === "string") return detail.message;
        try { return JSON.stringify(detail); } catch (_) { /* fall through */ }
    }
    if (typeof data?.message === "string" && data.message.trim()) return data.message;
    return `Backend request failed with status ${status}.`;
}

async function _fetch(url, options = {}) {
    const response = await fetch(url, options);
    let data = null;
    try { data = await response.json(); }
    catch (_) { throw new Error(`Backend returned an invalid response (HTTP ${response.status}).`); }
    if (!response.ok) throw new Error(extractErrorMessage(data, response.status));
    return data;
}

// ------------------------------------------------------------
// Health
// ------------------------------------------------------------

async function checkBackendHealth() {
    try {
        const r = await fetch(`${API_BASE_URL}/health`);
        return r.ok;
    } catch (_) { return false; }
}

// ------------------------------------------------------------
// PRIMARY: MINN Explain  (Objective 3 + 4 — demo patients)
// GET /api/minn/explain/demo/{patient_id}
// ------------------------------------------------------------

async function analyzeMINNDemo(patientId) {
    return _fetch(`${API_BASE_URL}/api/minn/explain/demo/${encodeURIComponent(patientId)}`);
}

// ------------------------------------------------------------
// PRIMARY: MINN Counterfactual  (Objective 4)
// GET /api/minn/counterfactual/demo/{patient_id}
// ------------------------------------------------------------

async function counterfactualMINNDemo(patientId) {
    return _fetch(`${API_BASE_URL}/api/minn/counterfactual/demo/${encodeURIComponent(patientId)}`);
}

// ------------------------------------------------------------
// PRIMARY: Flexible Upload-based analysis (1, 2, or 3 CSV files)
// POST /api/minn/explain/upload
// ------------------------------------------------------------

async function analyzeCrossModalUpload(patientId, files) {
    const formData = new FormData();
    if (patientId) formData.append("patient_id", patientId);
    if (files.genetic) formData.append("genetic_file", files.genetic);
    if (files.behavioral) formData.append("behavioral_file", files.behavioral);
    if (files.environmental) formData.append("environmental_file", files.environmental);
    return _fetch(`${API_BASE_URL}/api/minn/explain/upload`, { method: "POST", body: formData });
}

// ------------------------------------------------------------
// LEGACY: kept for backward-compat with Phase 07 callers
// POST /api/predict/all-three or POST /api/demo
// ------------------------------------------------------------

async function predictAllThree(patientId) {
    return _fetch(`${API_BASE_URL}/api/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: patientId, mode: "all_three" })
    });
}

async function analyzePatient(patientId, _mode) {
    // Default Analyze Patient workflow routes to the MINN endpoint for demo patients
    return analyzeMINNDemo(patientId);
}

// ------------------------------------------------------------
// Error formatter
// ------------------------------------------------------------

function getApiErrorMessage(error) {
    if (!error) return "An unknown backend error occurred.";
    if (error instanceof TypeError) return "Unable to connect to the backend. Make sure the FastAPI server is running.";
    return error.message || "An unexpected backend error occurred.";
}

// ------------------------------------------------------------
// Exports
// ------------------------------------------------------------

window.ASD_API = {
    API_BASE_URL,
    checkBackendHealth,
    analyzeMINNDemo,
    counterfactualMINNDemo,
    analyzeCrossModalUpload,
    predictAllThree,
    // legacy alias
    analyzePatient,
    getApiErrorMessage
};