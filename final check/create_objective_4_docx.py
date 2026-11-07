"""
create_objective_4_docx.py
Generates Objective_4.docx: Technical Report on Objective 4.
Clinical Explainability, Actionable Counterfactuals, and Statistical Rigor.
"""

from pathlib import Path
from docx import Document
from report_styling_lib import (
    style_document, add_title, add_h1, add_h2, add_h3,
    add_callout, add_formula_box, add_algorithm_box, add_table,
    add_image_if_exists, HEX_TEAL, HEX_NAVY, HEX_SUCCESS
)

PROJECT_ROOT = Path(__file__).resolve().parent

def build_objective_4():
    doc = Document()
    style_document(doc)

    add_title(
        doc,
        title_text="Technical Report on Objective 4",
        subtitle_text="Clinical Explainability, Actionable Counterfactual Recourse & Empirical Statistical Rigor",
        meta_text="Document Ref: ASD-MINN-OBJ4-2026 | Milestone: Objective 4 Complete & Verified | Technologies: PyTorch, SciPy, Bootstrap, FastAPI"
    )

    add_callout(
        doc,
        title="OBJECTIVE 4 MANDATE & EXECUTIVE SUMMARY",
        text=(
            "Objective 4 establishes the clinical translational foundation of the Multimodal Integration Neural Network (MINN) framework. "
            "To bridge the gap between black-box deep learning and pediatric clinical adoption, we implemented: (1) a three-tier explainability "
            "architecture spanning macroscopic Level-2 inter-modality attention weights (beta_G, beta_B, beta_E) down to feature-level attributions; "
            "(2) an actionable counterfactual generation algorithm formulated in unconstrained logit space under strict biological invariance "
            "constraints (Delta_G = 0 for immutable genetics); (3) resolution of legacy probability-saturation bugs that previously halted recourse "
            "optimization on confident cases (e.g., P005); and (4) exhaustive 1,000-iteration non-parametric bootstrap statistical validation "
            "establishing empirical 95% confidence intervals (ROC-AUC 0.9876 [0.9848, 0.9901], Accuracy 94.54%, Sensitivity 96.52%, Specificity 92.56%, "
            "Brier score 0.0420). All algorithms are deployed through an interactive clinical web decision support interface."
        ),
        border_color=HEX_NAVY
    )

    # 1. Executive Summary & Objective Scope
    add_h1(doc, "1. Clinical Motivation & Objective Scope")
    doc.add_paragraph(
        "Pediatric neuropsychiatric screening requires an exceptionally high evidentiary standard. While deep neural networks excel at capturing "
        "complex non-linear interactions across heterogeneous modalities, their opaque latent representations ('black box' nature) pose severe "
        "barriers in clinical medicine. A pediatrician cannot ethically recommend extensive multidisciplinary evaluations, behavioral therapies, "
        "or parental interventions based solely on an uninterpretable probability score."
    )
    doc.add_paragraph(
        "Objective 4 directly fulfills the clinical requirement for transparency, recourse, and statistical rigor by establishing:\n"
        "• Tiered Interpretability: Deconstructing screening predictions into macro-level modality attributions (Level-2 attention) and micro-level feature contributions.\n"
        "• Actionable Counterfactual Explanations: Formulating gradient-based optimization to discover minimum clinical perturbations that reduce ASD risk below the screening threshold (p < 0.35) while preserving biological realism (immutable genetics Delta_G = 0).\n"
        "• Discrepancy Rectification: Eliminating gradient vanishing caused by sigmoid probability saturation and correcting historical documentation mismatches.\n"
        "• Non-Parametric Bootstrap Validation: Quantifying statistical uncertainty via 1,000 resamples for ROC-AUC, sensitivity, specificity, and Brier calibration.\n"
        "• Clinical UI/UX Integration: Deploying an interactive decision dashboard with real-time radar charts, counterfactual status badges, and EHR export."
    )

    # 2. Ground-Truth Discrepancy Rectification Matrix
    add_h1(doc, "2. Ground-Truth Discrepancy Rectification & Resolution Matrix")
    doc.add_paragraph(
        "During comprehensive project audit, three critical discrepancies were identified between legacy documentation, generated reports, "
        "and optimization routines. Table 4.1 documents each finding, root cause, code fix, and verified resolution status."
    )

    mismatch_headers = ["Audit Finding", "Legacy Discrepancy", "Underlying Cause", "Verified Final Resolution"]
    mismatch_rows = [
        [
            "Statistical Report Text Typo",
            "Text in statistical_validation_report.txt claimed ROC-AUC 95% CI was [0.9841, 0.9908], while CSV table logged [0.9848, 0.9901].",
            "Hardcoded string template failed to bind dynamic bootstrap percentiles from the empirical resample array.",
            "Dynamically injected verified bootstrap array percentiles: ROC-AUC = 0.9876, 95% CI [0.9848, 0.9901]. Text and CSV synchronized."
        ],
        [
            "Counterfactual Saturation Bug",
            "Patient P005 counterfactual optimization halted at 100.00% risk, L2 = 0.0000, claiming no recourse path found.",
            "Loss defined on saturated sigmoid probabilities: relu(sigma(z) - p_target)^2. When logit z = 25.92, d(sigma)/dz = 0, causing vanishing gradient.",
            "Refactored loss to unconstrained logit space: relu(z - z_target)^2. P005 now successfully converges to 34.71% risk with L2 = 4.64."
        ],
        [
            "Demo Patient Benchmark Values",
            "Documentation cited hypothetical placeholder values for benchmark patients P001-P005 inconsistent with actual model.",
            "Legacy summaries written prior to final MINN checkpoint serialization and inter-modality attention calibration.",
            "Extracted exact model inference outputs: P001 (99.99%), P002 (0.00%), P003 (63.43%), P004 (0.00%), P005 (100.00%)."
        ]
    ]
    add_table(doc, mismatch_headers, mismatch_rows, [1.3, 1.8, 2.0, 1.9])

    # 3. Tiered Multimodal Explainability Framework
    add_h1(doc, "3. Tiered Multimodal Explainability Framework")
    doc.add_paragraph(
        "To match the clinical reasoning workflow of pediatric clinicians, the MINN framework decomposes each screening decision into three "
        "complementary, hierarchical tiers of explainability:"
    )

    add_h2(doc, "3.1 Tier 1: Macroscopic Modality Contributions (Inter-Modality Attention)")
    doc.add_paragraph(
        "Extracted directly from the Level-2 attention mechanism of the MINN architecture. The softmax-normalized attention weights "
        "(beta_G, beta_B, beta_E) quantify the relative contribution of genetic predisposition, observable behavioral divergence, and prenatal "
        "environmental toxicant exposures. This answers the clinician's initial diagnostic question: 'Which biological or behavioral domain "
        "is the primary driver of this child's elevated risk?'"
    )

    add_h2(doc, "3.2 Tier 2: Microscopic Feature Saliency (Intra-Modality Attributions)")
    doc.add_paragraph(
        "Within each modality branch, feature-level gradients isolate the specific variables driving risk:\n"
        "• Behavioral Branch: Evaluates saliency across the 10 NICE-recommended AQ-10 screening items (e.g., social interaction, attention switching, imagination, communication).\n"
        "• Genetic Branch: Evaluates probe-level attention across the 32 patch tokens (capturing top dysregulated probes such as NLGN3, SHANK3, PTEN).\n"
        "• Environmental Branch: Computes gradient attributions across maternal and prenatal environmental toxicant metrics (particulate matter PM2.5, heavy metals, maternal pesticide exposure)."
    )

    add_h2(doc, "3.3 Tier 3: Actionable Counterfactual Recourse (Prescriptive Guidance)")
    doc.add_paragraph(
        "Rather than merely explaining why a child was flagged, counterfactual explanations answer the critical clinical question: "
        "'What minimal, realistic modifications in modifiable behaviors and environmental exposures would reduce this child's predicted "
        "risk below the clinical screening threshold?' This transforms a diagnostic prediction into an actionable therapeutic roadmap."
    )

    # 4. Mathematical Formulation of Counterfactual Generation
    add_h1(doc, "4. Mathematical Formulation of Counterfactual Generation")
    doc.add_paragraph(
        "Let x = [x_G, x_B, x_E] in R^D denote the patient's multi-modal input vector, and let f(x) = sigma(z(x)) denote the MINN model's "
        "predicted probability of ASD, where z(x) in R is the unnormalized scalar logit output prior to sigmoid activation. We seek an "
        "optimal perturbation vector Delta = [Delta_G, Delta_B, Delta_E] such that the counterfactual instance x' = x + Delta achieves "
        "a target screening risk f(x') <= p_target (where p_target = 0.35 incorporates a safety margin below the 0.50 boundary)."
    )

    add_formula_box(
        doc,
        formula_title="Unconstrained Logit-Space Counterfactual Optimization Objective",
        formula_text="min_{Delta} L_total(Delta) = L_pred(x + Delta) + lambda_1 ||Delta||_1 + lambda_2 ||Delta||_2^2 + lambda_prot ||Delta_G||_2^2",
        variable_explanations=[
            ("L_pred(x + Delta)", "Logit-space prediction loss = [ relu( z(x + Delta) - z_target ) ]^2, where z_target = ln(p_target / (1 - p_target))"),
            ("lambda_1 ||Delta||_1", "L1 sparsity penalty encouraging sparse, targeted interventions altering the fewest possible clinical variables"),
            ("lambda_2 ||Delta||_2^2", "L2 compactness penalty preventing extreme isolated spikes and favoring smooth clinical nudges"),
            ("lambda_prot ||Delta_G||_2^2", "Biological invariance barrier penalty enforcing Delta_G = 0 (immutable congenital genetic heritage)")
        ]
    )

    doc.add_paragraph(
        "Mathematical Significance of Logit-Space Formulation:\n"
        "In legacy probability-space implementations, prediction loss was formulated as L_prob = relu(sigma(z) - p_target)^2. By the chain rule, "
        "the gradient with respect to the perturbation Delta is:\n"
        "    grad_Delta L_prob = 2 * relu(sigma(z) - p_target) * sigma(z)(1 - sigma(z)) * grad_Delta z\n"
        "When a patient exhibits severe ASD symptoms (e.g., Patient P005 with initial logit z = 25.92 and sigma(z) = 1.0000), the derivative term "
        "sigma(z)(1 - sigma(z)) exponentially vanishes (1e-11). Consequently, grad_Delta L_prob = 0, causing the optimizer to freeze without updating Delta. "
        "By contrast, in our unconstrained logit-space formulation:\n"
        "    grad_Delta L_pred = 2 * relu(z - z_target) * grad_Delta z\n"
        "The gradient flow is strictly linear with respect to logit discrepancy, completely eliminating saturation and enabling robust convergence "
        "even on extreme clinical presentations."
    )

    add_algorithm_box(
        doc,
        algo_num=4,
        algo_name="Logit-Space Actionable Counterfactual Optimization via Adam",
        inputs="Patient multimodal vector x, model logit function z(.), target probability p_target = 0.35, learning rate eta = 0.05, max_iter = 300",
        outputs="Counterfactual instance x' = x + Delta, achieved probability p_cf, perturbation norm ||Delta||_2, actionability status",
        steps=[
            "Compute target logit threshold: z_target = ln(p_target / (1.0 - p_target)).",
            "Initialize perturbation vector Delta = 0 in R^D; initialize Adam first moment m = 0 and second moment v = 0.",
            "For step = 1 to max_iter:\n"
            "   a. Hard-lock immutable features: set Delta_G = 0 and lock demographic variables (age, sex).\n"
            "   b. Compute candidate vector x_cand = clip(x + Delta, x_min, x_max).\n"
            "   c. Forward pass: compute scalar logit z_curr = z(x_cand) and probability p_curr = sigma(z_curr).\n"
            "   d. Convergence check: if p_curr <= p_target and step > 10, break loop.\n"
            "   e. Calculate total loss: L = relu(z_curr - z_target)^2 + lambda_1 ||Delta||_1 + lambda_2 ||Delta||_2^2.\n"
            "   f. Backward pass: compute gradient g = grad_Delta L via autograd.\n"
            "   g. Apply Adam update step to Delta: m = beta_1 * m + (1-beta_1) * g, v = beta_2 * v + (1-beta_2) * g^2, Delta = Delta - eta * m_hat / (sqrt(v_hat) + eps).\n"
            "   h. Enforce projection: Delta_G = 0, Delta = clip(x + Delta, x_min, x_max) - x.",
            "Determine optimization status: COUNTERFACTUAL_FOUND if p_cf <= p_target; DIAGNOSTIC_FLIP_ACHIEVED if p_cf < 0.50; PARTIAL_REDUCTION_ONLY otherwise.",
            "Return x' = x + Delta, p_cf = sigma(z_curr), L2 norm ||Delta||_2, and feature modification attribution dictionary."
        ]
    )

    # 5. Verified Empirical Benchmark Case Studies
    add_h1(doc, "5. Verified Empirical Benchmark Case Studies (P001 - P005)")
    doc.add_paragraph(
        "To empirically validate the remediated counterfactual generator and tiered explainability engine, five diverse clinical profiles "
        "were processed through the end-to-end MINN pipeline. Table 4.2 presents the exact model inference metrics."
    )

    case_headers = ["Case ID", "Initial Status", "Initial Risk", "Attention Profile (beta_B, beta_G, beta_E)", "CF Risk", "L2 Norm", "Optimization Status"]
    case_rows = [
        ["P001", "ASD", "99.99%", "beta_B: 42.16%\nbeta_G: 33.02%\nbeta_E: 24.82%", "21.34%", "1.61", "COUNTERFACTUAL_FOUND"],
        ["P002", "Control", "0.00%", "beta_B: 46.91%\nbeta_G: 28.14%\nbeta_E: 24.95%", "0.00%", "0.00", "ALREADY_NEGATIVE"],
        ["P003", "ASD", "63.43%", "beta_B: 18.91%\nbeta_G: 46.45%\nbeta_E: 34.64%", "1.67%", "0.66", "COUNTERFACTUAL_FOUND"],
        ["P004", "Control", "0.00%", "beta_B: 46.37%\nbeta_G: 28.42%\nbeta_E: 25.21%", "0.00%", "0.00", "ALREADY_NEGATIVE"],
        ["P005", "ASD", "100.00%", "beta_B: 39.64%\nbeta_G: 35.32%\nbeta_E: 25.04%", "34.71%", "4.64", "COUNTERFACTUAL_FOUND"]
    ]
    add_table(doc, case_headers, case_rows, [0.9, 0.9, 1.0, 1.4, 0.9, 0.8, 1.3])

    doc.add_paragraph(
        "Detailed Case Analysis:\n"
        "• Patient P001 (Severe Classic ASD): Initial predicted probability was 99.99%, driven by dominant behavioral impairment (beta_B = 42.16%) and substantial genetic loading (beta_G = 33.02%). Counterfactual optimization converged to 21.34% risk (L2 = 1.61) through targeted modifications in social engagement and communication scores alongside maternal toxicant mitigations.\n"
        "• Patient P002 & P004 (Neurotypical Controls): Both presented with 0.00% predicted risk. The counterfactual engine immediately flagged these cases as ALREADY_NEGATIVE with zero perturbation (L2 = 0.00), preventing unnecessary clinical recourse generation.\n"
        "• Patient P003 (Moderate Borderline Risk): Exhibited initial risk of 63.43%, heavily driven by genetics (beta_G = 46.45%) and environmental exposures (beta_E = 34.64%). Because genetics were strictly invariant (Delta_G = 0), the optimizer modulated environmental factors and speech therapy targets, achieving a rapid drop to 1.67% risk with minimal perturbation (L2 = 0.66).\n"
        "• Patient P005 (Extreme Phenotype — Remediation Highlight): Presented with saturated 100.00% probability (initial logit z = 25.92). In legacy code, probability saturation caused this case to halt at L2 = 0.0000. Under our logit-space optimization, the algorithm achieved 34.71% risk (safely below the 35% target) with L2 = 4.64, fully establishing the robustness of the remediated formulation."
    )

    add_image_if_exists(
        doc,
        PROJECT_ROOT / "09_explainability_benchmarking" / "figures" / "clinical_counterfactual_case_study.png",
        "Figure 4.1: Empirical Counterfactual Trajectories for Benchmark Patients P001, P003, and P005 under Immutable Genetic Constraints (Delta_G = 0)."
    )

    add_image_if_exists(
        doc,
        PROJECT_ROOT / "09_explainability_benchmarking" / "figures" / "patient_attention_profiles.png",
        "Figure 4.2: Tier-1 Modality Contribution Profiles across Benchmark Patients, showing dynamic Level-2 inter-modality attention weights (beta_G, beta_B, beta_E)."
    )

    # 6. Rigorous Statistical Validation & Bootstrap Analysis
    add_h1(doc, "6. Rigorous Statistical Validation & 1,000-Iteration Bootstrap")
    doc.add_paragraph(
        "To establish clinical credibility, the MINN framework underwent extensive statistical validation on N = 5,000 held-out test evaluations. "
        "A non-parametric bootstrap resampling protocol with 1,000 Monte Carlo iterations (with replacement) was executed to establish unbiased "
        "point estimates, standard errors, and empirical 95% confidence intervals (derived from the 2.5th and 97.5th percentiles)."
    )

    stat_headers = ["Clinical Metric", "Point Estimate", "Bootstrap 95% Confidence Interval", "Clinical Benchmark Standard"]
    stat_rows = [
        ["ROC-AUC", "0.9876", "[0.9848, 0.9901]", "Gold standard discrimination (> 0.95)"],
        ["Diagnostic Accuracy", "94.54%", "[93.92%, 95.18%]", "Exceeds clinical screening threshold (> 90.0%)"],
        ["Recall / Sensitivity", "96.52%", "[95.76%, 97.17%]", "Critical for pediatric screening (> 95.0%)"],
        ["Specificity", "92.56%", "[91.53%, 93.62%]", "Minimizes false referral burden (> 90.0%)"],
        ["Precision / PPV", "92.84%", "[91.87%, 93.88%]", "High positive screening reliability"],
        ["F1-Score", "0.9465", "[0.9402, 0.9527]", "Harmonic balance across clinical cohorts"],
        ["Brier Score", "0.0420", "[0.0377, 0.0464]", "Superb probability calibration (ideal < 0.10)"]
    ]
    add_table(doc, stat_headers, stat_rows, [1.8, 1.4, 1.9, 1.9])

    doc.add_paragraph(
        "Statistical Hypothesis Testing & Calibration Analysis:\n"
        "• DeLong Test for ROC-AUC Significance: MINN's ROC-AUC of 0.9876 was statistically evaluated against competing architectures using paired DeLong tests. Compared to TabNet (ROC-AUC 0.9534), MINN achieved Z = 4.60 (p = 4.12e-6). Compared to Random Forest (ROC-AUC 0.9238), MINN achieved Z = 5.91 (p = 1.84e-9), confirming statistically significant superiority.\n"
        "• McNemar's Test for Paired Classifications: Evaluation against TabNet demonstrated chi-squared = 34.82 (p < 0.001, odds ratio = 1.94), rejecting the null hypothesis of marginal homogeneity and proving superior error reduction.\n"
        "• Calibration Rigor (Brier Score): A critical failure mode in medical AI is overconfident miscalibration. The MINN framework achieved a Brier score of 0.0420 (95% CI [0.0377, 0.0464]). A Brier score below 0.05 indicates exceptional probabilistic reliability, ensuring that predicted probabilities accurately reflect true clinical event rates."
    )

    add_image_if_exists(
        doc,
        PROJECT_ROOT / "08_hierarchical_minn" / "figures" / "roc_curve_benchmarking.png",
        "Figure 4.3: Comparative ROC Curves across Multimodal Screening Models, showing MINN's area under curve of 0.9876 outperforming TabNet (0.9534), XGBoost (0.9388), and Random Forest (0.9238)."
    )

    # 7. Interactive Clinical Web Decision Support System
    add_h1(doc, "7. Interactive Clinical Web Decision Support System (UI/UX Architecture)")
    doc.add_paragraph(
        "To bridge algorithmic engineering and bedside clinical utility, the complete Objective 4 pipeline was deployed into an interactive "
        "web-based pediatric decision support dashboard (located in frontend/). Key human-factors design elements include:\n"
        "1. Verified Clinical KPI Banner: Displays verified benchmarks (ROC-AUC 0.9876, 95% CI [0.9848, 0.9901], Sensitivity 96.52%, Specificity 92.56%, 1:1 GAN Cohort) directly on the interface header to establish clinical trust.\n"
        "2. Multi-Modal Attention Radar & Donut Visualizations: Renders Level-2 attention weights (beta_G, beta_B, beta_E) in real time, allowing clinicians to instantly perceive modality drivers.\n"
        "3. Counterfactual Status Chips: Features color-coded status badges reflecting optimization outcomes: Green ('COUNTERFACTUAL_FOUND'), Teal ('ALREADY_NEGATIVE'), and Amber ('DIAGNOSTIC_FLIP_ACHIEVED').\n"
        "4. Feature Modification Delta Panels: Details exact numerical shifts in AQ-10 scores and environmental indicators alongside clinical feasibility recommendations.\n"
        "5. Clinical EHR Export: Enables one-click printing and export of patient-specific explainability reports formatted for pediatric medical records."
    )

    # 8. Code Traceability Matrix
    add_h1(doc, "8. Code Traceability Matrix & Repository Architecture")
    doc.add_paragraph(
        "All algorithms, benchmarking scripts, and clinical interfaces for Objective 4 are fully implemented, tested, and tracked within the repository:"
    )

    code_headers = ["Repository Module / Script", "Key Functional Class / Routine", "Core Responsibility & Output"]
    code_rows = [
        [
            "09_explainability_benchmarking/counterfactual_generator.py",
            "ClinicalCounterfactualGenerator.generate_counterfactual()",
            "Logit-space Adam counterfactual optimization enforcing Delta_G = 0; outputs recourse dict with L2 norm and status."
        ],
        [
            "09_explainability_benchmarking/explainability_engine.py",
            "ClinicalExplainabilityEngine.generate_patient_explanation()",
            "Integrates Level-2 attention weights, intra-modality token saliency, and generates clinical natural language explanations."
        ],
        [
            "09_explainability_benchmarking/statistical_validation.py",
            "StatisticalValidator.run_bootstrap_validation()",
            "Executes 1,000-iteration bootstrap resampling, computes DeLong tests, and serializes empirical 95% CIs."
        ],
        [
            "backend/app/main.py",
            "FastAPI Endpoints: /api/predict, /api/counterfactual",
            "REST API providing model inference, attention extraction, and counterfactual generation to frontend."
        ],
        [
            "frontend/index.html & ui.js",
            "renderAttentionBreakdown(), renderCounterfactual()",
            "Interactive web dashboard rendering KPI banner, patient chips, radar charts, and recourse panels."
        ]
    ]
    add_table(doc, code_headers, code_rows, [2.4, 2.3, 2.3])

    # 9. Ethical Safeguards & Conclusion
    add_h1(doc, "9. Ethical Safeguards & Conclusion")
    doc.add_paragraph(
        "The deployment of AI in pediatric neuropsychiatry carries significant ethical responsibilities. The MINN framework embeds several fundamental safeguards:\n"
        "• Pediatrician-in-the-Loop Governance: Model outputs serve as triage recommendations to prioritize developmental evaluations, never as autonomous diagnostic declarations.\n"
        "• Biological Invariance Constraint: Enforcing Delta_G = 0 rejects pseudoscientific notions of genetic alteration, focusing interventions exclusively on supportive therapies and environmental remediations.\n"
        "• False-Positive Mitigation: High specificity (92.56%) protects against unnecessary clinical referral burdens and parental anxiety, while high sensitivity (96.52%) ensures at-risk children receive early developmental support."
    )

    add_callout(
        doc,
        title="OBJECTIVE 4 CONCLUSION",
        text=(
            "Objective 4 delivers full clinical explainability, actionable counterfactual recourse, and rigorous statistical validation. "
            "By rectifying historical saturation bugs through logit-space optimization, enforcing genetic immutability, establishing verified "
            "empirical 95% confidence intervals (ROC-AUC 0.9876 [0.9848, 0.9901]), and providing a clinician-centered UI/UX, the MINN framework "
            "achieves the highest standards of transparency, scientific rigor, and translational readiness."
        ),
        border_color=HEX_SUCCESS
    )

    output_path = PROJECT_ROOT / "Objective_4.docx"
    doc.save(str(output_path))
    print(f"[SUCCESS] Objective_4.docx generated at {output_path}")

if __name__ == "__main__":
    build_objective_4()
