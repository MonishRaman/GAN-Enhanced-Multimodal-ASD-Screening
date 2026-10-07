/* =========================================================
   ASD MULTIMODAL ANALYSIS
   UI Rendering — Hierarchical Attention MINN (Objective 3 + 4)
   ========================================================= */

"use strict";

// ── DOM helpers ────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function firstEl(...ids) {
    for (const id of ids) { const el = $(id); if (el) return el; }
    return null;
}

function setText(id, val) {
    const el = $(id);
    if (el) el.textContent = val ?? "—";
}

function setVisible(id, show) {
    const el = $(id);
    if (!el) return;
    el.hidden = !show;
    el.style.display = show ? "" : "none";
}

function pct(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    const p = n * 100;
    return `${p < 1 ? p.toFixed(4) : p.toFixed(2)}%`;
}

// ── Backend status ─────────────────────────────────────────

function setBackendStatus(connected, message = null) {
    const text = firstEl("connectionText", "backendStatus");
    const dot  = $("connectionIndicator");

    if (text) {
        text.textContent = message || (connected ? "Backend connected" : "Backend not connected");
        text.classList.toggle("status-success", !!connected);
        text.classList.toggle("status-error",  connected === false);
    }
    if (dot) {
        dot.classList.toggle("online",  !!connected);
        dot.classList.toggle("offline", !connected);
    }
}

// ── Loading ────────────────────────────────────────────────

function showLoading(show = true) {
    const loading = firstEl("loadingSection");
    if (loading) { loading.hidden = !show; loading.style.display = show ? "" : "none"; }

    const btn = firstEl("analyzeButton");
    if (btn) {
        btn.disabled = show;
        if (show) {
            btn.dataset.origText = btn.textContent;
            btn.textContent = "Analyzing…";
        } else {
            btn.textContent = btn.dataset.origText || "Analyze Patient";
        }
    }
}

// ── Error ──────────────────────────────────────────────────

function showError(msg) {
    const el = firstEl("messageArea", "error-message", "errorMessage");
    if (!el) { console.error(msg); return; }
    el.textContent = msg;
    el.hidden = false;
    el.style.display = "";
    el.classList.add("error");
}

function clearError() {
    const el = firstEl("messageArea", "error-message", "errorMessage");
    if (!el) return;
    el.textContent = "";
    el.hidden = true;
    el.style.display = "none";
    el.classList.remove("error");
}

// ── Results visibility ─────────────────────────────────────

function showResults(show = true) {
    const el = firstEl("resultsSection");
    if (!el) return;
    el.hidden = !show;
    el.style.display = show ? "" : "none";
}

// ── Colour helpers ─────────────────────────────────────────

function barColour(weight) {
    // Map 0-1 weight to a blue→green gradient for attention bars
    const pct = Math.round(weight * 100);
    if (pct >= 50) return "#2563eb";   // dominant — primary blue
    if (pct >= 30) return "#7c3aed";   // moderate  — violet
    return "#64748b";                  // low       — slate
}

// ── Attention Weight Bar ───────────────────────────────────

function attentionBar(label, weight, isPrimary) {
    const pct100 = Math.min(100, Math.max(0, weight * 100)).toFixed(1);
    const col    = isPrimary ? "#2563eb" : "#94a3b8";
    return `
    <div class="att-row ${isPrimary ? "att-primary" : ""}">
        <div class="att-label">
            <span>${label}</span>
            <span class="att-pct">${pct100}%</span>
        </div>
        <div class="att-track">
            <div class="att-fill" style="width:${pct100}%; background:${col};"></div>
        </div>
    </div>`;
}

// ── Saliency mini-bars (top-5 dimensions) ─────────────────

function saliencyRows(dimArr) {
    if (!Array.isArray(dimArr) || dimArr.length === 0) {
        return `<p class="muted-text">No saliency data.</p>`;
    }
    const max = Math.max(...dimArr.map(d => d.attention_weight), 0.001);
    return dimArr.map(d => {
        const w = Math.min(100, (d.attention_weight / max) * 100).toFixed(1);
        return `
        <div class="sal-row">
            <span class="sal-dim">Dim ${d.dimension}</span>
            <div class="sal-track">
                <div class="sal-fill" style="width:${w}%"></div>
            </div>
            <span class="sal-val">${d.attention_weight.toFixed(4)}</span>
        </div>`;
    }).join("");
}

// ── Counterfactual render ──────────────────────────────────

function renderCounterfactual(cfData) {
    const section = $("cfSection");
    const body    = $("cfBody");
    if (!section || !body) return;

    if (!cfData || cfData.status === "error") {
        section.hidden = true;
        return;
    }

    const cf = cfData.counterfactual || {};
    let html = "";

    if (cf.status === "ALREADY_NEGATIVE") {
        const origPct = typeof cf.original_probability === "number"
            ? (cf.original_probability * 100).toFixed(2)
            : "0.00";
        html += `
        <div class="cf-status cf-ok">
            ✓ ${cf.message || `Patient is screened as Control / Non-ASD (${origPct}% risk). No counterfactual required.`}
        </div>`;
    } else if (cf.status === "COUNTERFACTUAL_FOUND") {
        const origDiag = cf.original_diagnosis || {};
        const outcome  = cf.counterfactual_outcome || {};
        const cost     = cf.perturbation_cost || {};
        const recs     = cf.actionable_recommendations || {};

        const origProbPct   = typeof origDiag.probability === "number" ? (origDiag.probability * 100).toFixed(2) : "—";
        const cfProbPct     = typeof outcome.probability === "number" ? (outcome.probability * 100).toFixed(2) : "—";
        const reductionPct  = typeof outcome.probability_reduction === "number" ? (outcome.probability_reduction * 100).toFixed(2) : "—";
        const targetThresh  = typeof outcome.target_threshold === "number" ? (outcome.target_threshold * 100).toFixed(0) : "35";

        html += `
        <div class="cf-status cf-ok">
            ✓ Actionable Counterfactual Found — Target Risk ≤ ${targetThresh}% Achieved (${outcome.classification || "Control"})
        </div>

        <div class="cf-metric-row">
            <span>Original ASD Risk</span>
            <strong>${origProbPct}% (${origDiag.classification || "ASD"})</strong>
        </div>
        <div class="cf-metric-row">
            <span>Counterfactual Risk Outcome</span>
            <strong style="color: var(--success);">${cfProbPct}%</strong>
        </div>
        <div class="cf-metric-row">
            <span>Probability Reduction (Δ)</span>
            <strong style="color: var(--primary);">−${reductionPct}%</strong>
        </div>
        <div class="cf-metric-row">
            <span>Intervention Cost (L₂ Distance)</span>
            <strong>${typeof cost.l2_distance === "number" ? cost.l2_distance.toFixed(4) : "—"}</strong>
        </div>
        <div class="cf-metric-row">
            <span>Sparsity Cost (L₁ Distance)</span>
            <strong>${typeof cost.l1_distance === "number" ? cost.l1_distance.toFixed(4) : "—"}</strong>
        </div>`;

        if (cf.clinical_summary) {
            html += `
            <div style="margin-top: 12px; padding: 10px 12px; background: var(--amber-soft); border-left: 3px solid var(--amber); border-radius: var(--radius-sm); font-size: 13px; line-height: 1.5; color: var(--text-2);">
                <strong>Clinical Summary:</strong> ${cf.clinical_summary}
            </div>`;
        }

        const bRecs = Array.isArray(recs.behavioral_adjustments) ? recs.behavioral_adjustments : [];
        const eRecs = Array.isArray(recs.environmental_adjustments) ? recs.environmental_adjustments : [];

        if (bRecs.length > 0 || eRecs.length > 0) {
            html += `<div class="cf-recs-title">Actionable Interventions by Modality</div>`;

            bRecs.forEach(item => {
                const shiftSign = item.recommended_shift > 0 ? "+" : "";
                html += `
                <div class="cf-rec">
                    <span class="cf-modality-tag" style="background: #d1fae5; color: #065f46;">Behavioral</span>
                    <span class="cf-rec-text">
                        Adjust <strong>Dimension ${item.dimension}</strong> by <strong>${shiftSign}${item.recommended_shift.toFixed(4)}</strong> (actionable behavioral therapy target)
                    </span>
                </div>`;
            });

            eRecs.forEach(item => {
                const shiftSign = item.recommended_shift > 0 ? "+" : "";
                html += `
                <div class="cf-rec">
                    <span class="cf-modality-tag" style="background: #ede9fe; color: #5b21b6;">Environmental</span>
                    <span class="cf-rec-text">
                        Adjust <strong>Dimension ${item.dimension}</strong> by <strong>${shiftSign}${item.recommended_shift.toFixed(4)}</strong> (actionable environmental exposure modification)
                    </span>
                </div>`;
            });
        }
    } else {
        html = `<p class="muted-text">Counterfactual analysis returned status: ${cf.status || "UNKNOWN"}</p>`;
    }

    body.innerHTML = html;
    section.hidden = false;
}

// ── Main MINN & Cross-modal result renderer ────────────────

const DEMO_PATIENTS = ["P001", "P002", "P003", "P004", "P005"];

function renderAnalysisResult(data) {
    if (!data) return;
    clearError();
    showResults(true);

    // Support both: MINN schema { status, result: {...} } and Phase 07 upload { cross_modal: {...}, modality_results: [...] }
    const result       = data.result || data;
    const isCrossModal = !result.prediction && !!(result.cross_modal || data.cross_modal);
    const cm           = result.cross_modal || data.cross_modal || {};

    const pred   = isCrossModal ? cm : (result.prediction || {});
    const att2   = result.modality_attention_level_2 || {};
    const sal    = result.feature_saliency_level_1   || {};

    const patientId  = result.patient_id || data.patient_id || "uploaded-patient";
    const asdProb    = pred.asd_probability ?? pred.probability ?? null;
    const classif    = pred.classification  || pred.prediction  || (asdProb !== null ? (asdProb >= 0.5 ? "ASD" : "Control") : "—");
    const confidence = pred.confidence ?? (asdProb !== null ? (asdProb >= 0.5 ? asdProb : (1 - asdProb)) : null);
    const threshold  = pred.decision_threshold ?? pred.threshold ?? 0.50;
    const modelName  = isCrossModal
        ? (cm.model || "Cross-Modal Multimodal Fusion Network")
        : "Hierarchical Attention MINN (Objective 3)";

    // ── Patient + prediction header ──────────────────────
    setText("resultPatient", `Patient: ${patientId}`);
    setText("resultStatus",  classif === "ASD" ? "ASD" : "Control");

    // Colour result status badge
    const statusEl = $("resultStatus");
    if (statusEl) {
        statusEl.classList.toggle("badge-asd",     classif === "ASD");
        statusEl.classList.toggle("badge-non-asd", classif !== "ASD");
    }

    setText("finalPrediction",  classif);
    setText("finalProbability", pct(asdProb));
    setText("finalThreshold",   threshold.toFixed ? threshold.toFixed(2) : "0.50");
    setText("finalConfidence",  confidence !== null ? pct(confidence) : "—");
    setText("modelLabel",       modelName);

    // colour final prediction text
    const predEl = $("finalPrediction");
    if (predEl) {
        predEl.classList.toggle("prediction-asd",     classif === "ASD");
        predEl.classList.toggle("prediction-non-asd", classif !== "ASD");
    }

    // Probability bar
    const bar = $("finalProbabilityBar");
    if (bar && asdProb !== null) {
        const w = Math.max(0, Math.min(100, asdProb * 100)).toFixed(2);
        bar.style.width = `${w}%`;
        bar.style.background = classif === "ASD" ? "#dc2626" : "#15803d";
    }

    // ── Clinical interpretation ───────────────────────────
    let clinicalText = result.clinical_interpretation || "";
    if (!clinicalText && isCrossModal && Array.isArray(data.modality_results)) {
        const gMod = data.modality_results.find(m => m.modality === "genetic") || {};
        const bMod = data.modality_results.find(m => m.modality === "behavioral") || {};
        const eMod = data.modality_results.find(m => m.modality === "environmental") || {};
        clinicalText = `Patient ${patientId} is screened as ${classif} with ${pct(confidence)} confidence using the ${modelName}. Individual modality expert screening: Genetic (${pct(gMod.probability)}), Behavioral (${pct(bMod.probability)}), Environmental (${pct(eMod.probability)}). Joint 32-D latent alignment integrated all three representations.`;
    }

    setText("clinicalInterpretation", clinicalText);
    setVisible("clinicalSection", !!clinicalText);

    // ── Level-2 Modality Attention weights / Modality Results ──
    const attContainer = $("attentionWeights");
    if (attContainer && Object.keys(att2).length > 0) {
        const gw  = att2.genetic_weight       ?? 0;
        const bw  = att2.behavioral_weight    ?? 0;
        const ew  = att2.environmental_weight ?? 0;
        const pdm = att2.primary_driving_modality || "";

        attContainer.innerHTML = `
            ${attentionBar("Genetic (β_G)",       gw,  pdm === "Genetic")}
            ${attentionBar("Behavioral (β_B)",    bw,  pdm === "Behavioral")}
            ${attentionBar("Environmental (β_E)", ew,  pdm === "Environmental")}
            <p class="att-note">
                Primary driving modality: <strong>${pdm || "—"}</strong> (${pdm === "Genetic" ? pct(gw) : pdm === "Behavioral" ? pct(bw) : pct(ew)} dynamic weight)
            </p>`;
        setVisible("attentionSection", true);
    } else if (attContainer && isCrossModal && Array.isArray(data.modality_results)) {
        const gMod = data.modality_results.find(m => m.modality === "genetic") || {};
        const bMod = data.modality_results.find(m => m.modality === "behavioral") || {};
        const eMod = data.modality_results.find(m => m.modality === "environmental") || {};

        attContainer.innerHTML = `
            ${attentionBar("Genetic Modality Expert",       gMod.probability ?? 0, (gMod.probability ?? 0) >= 0.5)}
            ${attentionBar("Behavioral Modality Expert",    bMod.probability ?? 0, (bMod.probability ?? 0) >= 0.5)}
            ${attentionBar("Environmental Modality Expert", eMod.probability ?? 0, (eMod.probability ?? 0) >= 0.5)}
            <p class="att-note">
                Cross-modal fusion: <strong>32-D Joint Latent Space</strong> · Preprocessing &amp; Encoding: <strong>3-Modality Aligned</strong>
            </p>`;
        setVisible("attentionSection", true);
    } else {
        setVisible("attentionSection", false);
    }

    // ── Level-1 Feature Saliency ──────────────────────────
    const salG = $("saliencyGenetic");
    const salB = $("saliencyBehavioral");
    const salE = $("saliencyEnvironmental");

    if (salG) salG.innerHTML = saliencyRows(sal.top_genetic_dimensions);
    if (salB) salB.innerHTML = saliencyRows(sal.top_behavioral_dimensions);
    if (salE) salE.innerHTML = saliencyRows(sal.top_environmental_dimensions);

    const hasSal = !!(sal.top_genetic_dimensions || sal.top_behavioral_dimensions);
    setVisible("saliencySection", hasSal);

    // Raw JSON
    const raw = $("rawResponse");
    if (raw) raw.textContent = JSON.stringify(data, null, 2);

    // ── Trigger counterfactual fetch ──────────────────────
    const cfSection     = $("cfSection");
    const cfBody        = $("cfBody");
    const isDemoPatient = DEMO_PATIENTS.includes((patientId || "").trim().toUpperCase());

    if (data.counterfactual) {
        // Counterfactual was computed directly by the MINN upload endpoint!
        renderCounterfactual(data);
    } else if (isDemoPatient && window.ASD_API && typeof window.ASD_API.counterfactualMINNDemo === "function") {
        if (cfSection && cfBody) {
            cfSection.hidden = false;
            cfBody.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px; padding: 12px; color: var(--muted); font-size: 13px;">
                    <div class="spinner" style="width: 18px; height: 18px; border-width: 2px;"></div>
                    <span>Computing gradient-based counterfactual recommendations…</span>
                </div>`;
        }

        window.ASD_API.counterfactualMINNDemo(patientId)
            .then(cf => renderCounterfactual(cf))
            .catch(err => {
                if (cfBody) {
                    cfBody.innerHTML = `<p class="muted-text">Counterfactual analysis unavailable: ${err.message || err}</p>`;
                }
            });
    } else if (cfSection && cfBody) {
        cfSection.hidden = false;
        cfBody.innerHTML = `
            <div class="cf-status cf-ok" style="background: var(--primary-soft); color: var(--text);">
                ✓ <strong>Custom CSV Upload Processed:</strong> 3-modality representation screened via Hierarchical Attention MINN.
            </div>`;
    }
}

// ── Clear UI ───────────────────────────────────────────────

function clearAnalysisUI() {
    clearError();
    showLoading(false);
    showResults(false);

    setText("resultPatient",         "Patient: —");
    setText("resultStatus",          "—");
    setText("finalPrediction",       "—");
    setText("finalProbability",      "—");
    setText("finalThreshold",        "0.50");
    setText("finalConfidence",       "—");
    setText("modelLabel",            "—");
    setText("clinicalInterpretation","");
    setVisible("clinicalSection",    false);
    setVisible("attentionSection",   false);
    setVisible("saliencySection",    false);
    setVisible("cfSection",          false);

    const bar = $("finalProbabilityBar");
    if (bar) { bar.style.width = "0%"; bar.style.background = "#2563eb"; }

    const raw = $("rawResponse");
    if (raw) raw.textContent = "—";
}

// ── Init ───────────────────────────────────────────────────

function initializeUI() {
    clearError();
    showLoading(false);
    showResults(false);
    console.log("MINN UI layer initialized.");
}

// ── Exports ────────────────────────────────────────────────

window.ASD_UI = {
    setBackendStatus,
    showLoading,
    showError,
    clearError,
    showResults,
    renderAnalysisResult,
    clearAnalysisUI,
    initializeUI,
    renderCounterfactual
};

document.addEventListener("DOMContentLoaded", initializeUI);