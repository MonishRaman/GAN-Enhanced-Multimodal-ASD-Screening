"""
Objective 3: Hierarchical Attention-Based Multi-Input Neural Network (MINN)
for Multimodal ASD Screening.

Architecture:
- 3 Dedicated Modality Branches (Genetic, Behavioral, Environmental)
- Level 1: Intra-Modality Feature Attention (learns feature saliency per modality)
- Level 2: Inter-Modality Dynamic Attention (learns patient-specific modality importance)
- Adaptive Feature Fusion
- ASD Classifier with Dropout & Sigmoid probability output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityBranch(nn.Module):
    """Learns branch-level representations for a single modality."""
    def __init__(self, in_features=32, hidden_dim=64, out_dim=32, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.encoder(x)

class IntraModalityAttention(nn.Module):
    """
    Level 1 Attention: Learns which features within a modality are salient.
    Computes an attention distribution over feature dimensions.
    """
    def __init__(self, feature_dim=32):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, h):
        # h: (batch_size, feature_dim)
        scores = self.attention_net(h)  # (batch_size, feature_dim)
        alpha = F.softmax(scores, dim=-1) # feature-level attention weights
        # Reweight features with a residual connection
        h_weighted = (1.0 + alpha) * h
        return h_weighted, alpha

class InterModalityAttention(nn.Module):
    """
    Level 2 Attention: Dynamically learns relative modality importance (beta_G, beta_B, beta_E)
    for each individual patient, replacing static fixed weights.
    """
    def __init__(self, feature_dim=32, hidden_dim=16):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_g, h_b, h_e):
        # Compute scalar relevance score for each modality per sample
        s_g = self.scorer(h_g)  # (batch_size, 1)
        s_b = self.scorer(h_b)  # (batch_size, 1)
        s_e = self.scorer(h_e)  # (batch_size, 1)

        scores = torch.cat([s_g, s_b, s_e], dim=1) # (batch_size, 3)
        beta = F.softmax(scores, dim=1)           # (batch_size, 3): [beta_G, beta_B, beta_E]
        return beta

class HierarchicalAttentionMINN(nn.Module):
    """
    Complete Hierarchical Attention-Based Multi-Input Neural Network.
    Integrates Genetic, Behavioral, and Environmental representations.
    """
    def __init__(self, in_features=32, branch_hidden=64, branch_out=32, classifier_dropout=0.3):
        super().__init__()
        self.in_features = in_features

        # 1. Modality-Specific Feature Learning Branches
        self.genetic_branch = ModalityBranch(in_features, branch_hidden, branch_out)
        self.behavioral_branch = ModalityBranch(in_features, branch_hidden, branch_out)
        self.environmental_branch = ModalityBranch(in_features, branch_hidden, branch_out)

        # 2. Level 1: Intra-Modality Feature Attention
        self.intra_genetic = IntraModalityAttention(branch_out)
        self.intra_behavioral = IntraModalityAttention(branch_out)
        self.intra_environmental = IntraModalityAttention(branch_out)

        # 3. Level 2: Inter-Modality Attention
        self.inter_attention = InterModalityAttention(branch_out)

        # 4. Multimodal Fusion & Classification Head
        # Concatenation of 3 attention-weighted modalities: 3 * branch_out
        fusion_dim = branch_out * 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, g, b, e, return_attention=False):
        # 1. Branch encoding
        h_g = self.genetic_branch(g)
        h_b = self.behavioral_branch(b)
        h_e = self.environmental_branch(e)

        # 2. Intra-modality feature attention
        h_g_att, alpha_g = self.intra_genetic(h_g)
        h_b_att, alpha_b = self.intra_behavioral(h_b)
        h_e_att, alpha_e = self.intra_environmental(h_e)

        # 3. Inter-modality dynamic attention
        beta = self.inter_attention(h_g_att, h_b_att, h_e_att) # (batch, 3)
        beta_g = beta[:, 0:1]
        beta_b = beta[:, 1:2]
        beta_e = beta[:, 2:3]

        # 4. Adaptive multimodal fusion
        fused = torch.cat([
            beta_g * h_g_att,
            beta_b * h_b_att,
            beta_e * h_e_att
        ], dim=1) # (batch, 96)

        # 5. Classification
        logits = self.classifier(fused).squeeze(1) # (batch,)
        probs = torch.sigmoid(logits)

        if return_attention:
            attention_dict = {
                "intra_genetic": alpha_g,
                "intra_behavioral": alpha_b,
                "intra_environmental": alpha_e,
                "inter_modality": beta # columns: [Genetic, Behavioral, Environmental]
            }
            return logits, probs, attention_dict

        return logits, probs
