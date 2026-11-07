# Methodological Justification and Limitations of Distance Thresholds

**Audit Date:** 2026-10-30 16:35:08  
**Target:** Representation Separation and Leakage Defense  

## 1. Role and Justification of the 0.05 Heuristic Threshold
The minimum Euclidean distance threshold of 0.05 is an empirical heuristic applied to 32-dimensional continuous embeddings to verify that representations of opposing classes (ASD vs Control) do not collapse onto identical or near-identical coordinates in the shared latent space. In neural networks with non-linear bottleneck alignments, dimension collapse can occur when different classes are mapped to overlapping regions, leading to severe miscalibration.

The observed minimum cross-label distance of **0.0738** demonstrates that the learned 32-D manifold maintains separation between classes.

## 2. Critical Methodological Limitation: What Distance Thresholds Do NOT Prove
- **A distance threshold does NOT independently prove the absence of data leakage.**
- Distance $> 0.05$ simply confirms geometric separation; a model could theoretically overfit or leak features while maintaining separated representations.
- **Proof of zero data leakage rests entirely on partition hygiene:**
  1. **Disjoint Record Partitioning:** Zero shared record identifiers between training and testing sets across all three modalities ($I_\text{train} \cap I_\text{test} = \emptyset$).
  2. **Zero Synthetic Test Contamination:** Confining all 46 WGAN-GP synthetic minority control vectors to the training partition (test set contains 0 synthetic vectors).
  3. **Zero Identical Records:** Exact duplicate checks confirm zero identical feature vectors across train and test partitions.
  4. **Pipeline Isolation:** Scalers and encoders fit solely on training data and frozen before evaluation.

## 3. Generalizability Limitation
Neither geometric distance separation nor strict data partition hygiene proves that the model will generalize to live pediatric clinical screening populations. Performance on in-silico assembled combinations demonstrates mathematical architecture functionality under idealized conditions, not prospective clinical effectiveness.
