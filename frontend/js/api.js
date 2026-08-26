// ============================================================
// ASD MULTIMODAL ANALYSIS
// Frontend -> Backend API Communication
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";

function extractErrorMessage(data, status) {
    const detail = data?.detail;

    if (typeof detail === "string" && detail.trim()) {
        return detail;
    }

    if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0];

        if (typeof first === "string") {
            return first;
        }

        if (first && typeof first.msg === "string") {
            return first.msg;
        }
    }

    if (detail && typeof detail === "object") {
        if (typeof detail.message === "string") {
            return detail.message;
        }

        try {
            return JSON.stringify(detail);
        } catch (error) {
            // Fall through to generic message.
        }
    }

    if (typeof data?.message === "string" && data.message.trim()) {
        return data.message;
    }

    return `Backend request failed with status ${status}.`;
}

/**
 * Check whether the FastAPI backend is reachable.
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: "GET"
        });

        return response.ok;
    } catch (error) {
        console.error("Backend health check failed:", error);
        return false;
    }
}


/**
 * Send a patient analysis request to the FastAPI backend.
 *
 * @param {string} patientId
 * @param {string} mode
 * @returns {Promise<Object>}
 */
async function analyzePatient(patientId, mode) {

    const response = await fetch(`${API_BASE_URL}/api/demo`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            patient_id: patientId,
            mode: mode
        })
    });

    let data = null;

    try {
        data = await response.json();
    } catch (error) {
        throw new Error(
            `Backend returned an invalid response (HTTP ${response.status}).`
        );
    }

    if (!response.ok) {
        const message = extractErrorMessage(data, response.status);

        throw new Error(message);
    }

    return data;
}


/**
 * Run general inference using explicit modality file paths.
 *
 * @param {string} patientId
 * @param {string} mode
 * @param {{genetic?: string, behavioral?: string, environmental?: string}} filePaths
 * @returns {Promise<Object>}
 */
async function inferPatient(patientId, mode, filePaths = {}) {

    const payload = {
        patient_id: patientId,
        mode: mode
    };

    if (filePaths.genetic) {
        payload.genetic = {
            modality: "genetic",
            file_path: filePaths.genetic
        };
    }

    if (filePaths.behavioral) {
        payload.behavioral = {
            modality: "behavioral",
            file_path: filePaths.behavioral
        };
    }

    if (filePaths.environmental) {
        payload.environmental = {
            modality: "environmental",
            file_path: filePaths.environmental
        };
    }

    const response = await fetch(`${API_BASE_URL}/api/infer`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });

    let data = null;

    try {
        data = await response.json();
    } catch (error) {
        throw new Error(
            `Backend returned an invalid response (HTTP ${response.status}).`
        );
    }

    if (!response.ok) {
        const message = extractErrorMessage(data, response.status);

        throw new Error(message);
    }

    return data;
}


window.ASD_API = {
    API_BASE_URL,
    checkBackendHealth,
    analyzePatient,
    inferPatient,
    getApiErrorMessage
};


/**
 * Convert backend errors into a user-friendly message.
 */
function getApiErrorMessage(error) {

    if (!error) {
        return "An unknown backend error occurred.";
    }

    if (error instanceof TypeError) {
        return "Unable to connect to the backend. Make sure the FastAPI server is running.";
    }

    return error.message || "An unexpected backend error occurred.";
}