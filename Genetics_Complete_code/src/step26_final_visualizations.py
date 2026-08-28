import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 26 — FINAL RESULTS VISUALIZATION
# ============================================================

BASE = r"D:\org genetical data"

OUTPUT = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "final_visualizations"
)

os.makedirs(OUTPUT, exist_ok=True)


# ============================================================
# 1. MODEL COMPARISON
# ============================================================

model_data = {
    "Model": [
        "Logistic Regression",
        "Linear SVM",
        "Random Forest"
    ],

    "Accuracy": [
        0.6471,
        0.7507,
        0.7504
    ],

    "Balanced Accuracy": [
        0.6784,
        0.6050,
        0.5475
    ],

    "F1": [
        0.7194,
        0.8407,
        0.8516
    ],

    "ROC-AUC": [
        0.7501,
        0.7152,
        0.6818
    ]
}

models = pd.DataFrame(model_data)

models.to_csv(
    os.path.join(
        OUTPUT,
        "model_comparison_final.csv"
    ),
    index=False
)


# ============================================================
# ROC-AUC MODEL COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    models["Model"],
    models["ROC-AUC"]
)

plt.ylabel("ROC-AUC")
plt.title("Classical Model ROC-AUC Comparison")
plt.ylim(0, 1)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT,
        "model_roc_auc_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. FEATURE-SIZE EXPERIMENT
# ============================================================

feature_sizes = pd.DataFrame({

    "Features": [
        25,
        50,
        100,
        250,
        500,
        1000
    ],

    "Accuracy": [
        0.698188,
        0.655435,
        0.681522,
        0.707609,
        0.647101,
        0.647101
    ],

    "Balanced_Accuracy": [
        0.683856,
        0.676013,
        0.708105,
        0.696732,
        0.634641,
        0.678431
    ],

    "F1": [
        0.779684,
        0.733695,
        0.754735,
        0.785164,
        0.735985,
        0.719380
    ],

    "ROC_AUC": [
        0.716601,
        0.736035,
        0.768431,
        0.768039,
        0.756536,
        0.750109
    ]
})


feature_sizes.to_csv(
    os.path.join(
        OUTPUT,
        "feature_size_final.csv"
    ),
    index=False
)


# ============================================================
# FEATURE SIZE VS ROC-AUC
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    feature_sizes["Features"],
    feature_sizes["ROC_AUC"],
    marker="o"
)

plt.xlabel("Number of Selected Features")
plt.ylabel("Mean ROC-AUC")
plt.title("Feature Size vs ROC-AUC")
plt.ylim(0.5, 0.85)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT,
        "feature_size_vs_roc_auc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FEATURE SIZE VS BALANCED ACCURACY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    feature_sizes["Features"],
    feature_sizes["Balanced_Accuracy"],
    marker="o"
)

plt.xlabel("Number of Selected Features")
plt.ylabel("Balanced Accuracy")
plt.title("Feature Size vs Balanced Accuracy")
plt.ylim(0.4, 0.8)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT,
        "feature_size_vs_balanced_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. STATISTICAL VALIDATION
# ============================================================

statistics_file = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "statistical_validation",
    "stable_features_statistical_validation.csv"
)

stats = pd.read_csv(
    statistics_file
)

stats["Feature_Label"] = (
    stats["Probe"]
    .astype(str)
)

stats["minus_log10_FDR"] = -np.log10(
    stats["FDR_p_value"].clip(
        lower=1e-300
    )
)


# ============================================================
# STATISTICAL FEATURE PLOT
# ============================================================

plot_stats = stats.sort_values(
    "FDR_p_value"
)

plt.figure(figsize=(11, 7))

plt.barh(
    plot_stats["Feature_Label"],
    plot_stats["minus_log10_FDR"]
)

plt.axvline(
    -np.log10(0.05),
    linestyle="--",
    label="FDR = 0.05"
)

plt.xlabel("-log10(FDR)")
plt.ylabel("Stable Probe")
plt.title(
    "Statistical Significance of Stable Features"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT,
        "stable_feature_statistical_significance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. INTEGRATED EVIDENCE
# ============================================================

integrated_file = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "integrated_evidence",
    "integrated_stable_feature_evidence.csv"
)

integrated = pd.read_csv(
    integrated_file
)

integrated = integrated.sort_values(
    "Evidence_Score",
    ascending=True
)


plt.figure(figsize=(11, 7))

plt.barh(
    integrated["Probe"].astype(str),
    integrated["Evidence_Score"]
)

plt.xlabel("Evidence Score")
plt.ylabel("Probe")
plt.title(
    "Integrated Evidence Ranking of Stable Features"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT,
        "integrated_evidence_ranking.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. FINAL CANDIDATE GENE TABLE
# ============================================================

biological_file = os.path.join(
    BASE,
    "genetics",
    "outputs",
    "biological_interpretation",
    "biological_interpretation_stable_genes.csv"
)

biological = pd.read_csv(
    biological_file
)

biological.to_csv(
    os.path.join(
        OUTPUT,
        "final_candidate_genes.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT,
    "visualization_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 26 — FINAL VISUALIZATION SUMMARY\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )

    f.write(
        "Dataset\n"
    )

    f.write(
        "116 samples\n"
        "87 ASD\n"
        "29 Control\n\n"
    )

    f.write(
        "Feature selection\n"
    )

    f.write(
        "33,357 filtered probes\n"
        "1,000 MI-selected features\n"
        "13 stable features selected in >=3 folds\n\n"
    )

    f.write(
        "Best feature-size ROC-AUC\n"
    )

    f.write(
        "100 features: 0.7684\n\n"
    )

    f.write(
        "Best feature-size accuracy\n"
    )

    f.write(
        "250 features: 0.7076\n\n"
    )

    f.write(
        "Best classical ROC-AUC\n"
    )

    f.write(
        "Logistic Regression: 0.7501\n\n"
    )

    f.write(
        "Statistical validation\n"
    )

    f.write(
        "7 nominally significant features\n"
        "4 FDR-significant features\n\n"
    )

    f.write(
        "Primary biological candidate\n"
    )

    f.write(
        "FGFR1\n\n"
    )

    f.write(
        "Other annotated candidates\n"
    )

    f.write(
        "UBAP1\n"
        "SLC48A1\n"
        "BDH1\n"
        "KLK10\n"
        "GALE\n"
    )


print()
print("=" * 70)
print("STEP 26 — FINAL VISUALIZATION")
print("=" * 70)

print()
print("Generated files:")
print()

for file in sorted(
    os.listdir(OUTPUT)
):

    print(
        os.path.join(
            OUTPUT,
            file
        )
    )

print()
print(
    "Step 26 completed successfully."
)