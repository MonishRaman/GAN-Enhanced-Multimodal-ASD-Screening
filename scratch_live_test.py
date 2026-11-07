import sys
sys.path.append('08_hierarchical_minn')
sys.path.append('09_explainability_benchmarking')
import torch
from models.hierarchical_minn import HierarchicalAttentionMINN
from counterfactual_generator import CounterfactualGenerator
import pandas as pd

device = torch.device('cpu')
model = HierarchicalAttentionMINN(in_features=32, branch_hidden=64, branch_out=32).to(device)
model.load_state_dict(torch.load('08_hierarchical_minn/models/minn_best.pt', map_location=device))
model.eval()

gen_df = pd.read_csv('08_hierarchical_minn/data/genetic_32d_test_unseen.csv')
beh_df = pd.read_csv('08_hierarchical_minn/data/behavioral_32d_all.csv')
env_df = pd.read_csv('08_hierarchical_minn/data/environmental_32d_all.csv')

def get_row(df, idx):
    feat_cols = [c for c in df.columns if c not in ['Target', 'Class/ASD', 'Sample_Index', 'GSM_ID', 'Is_Synthetic']]
    return torch.tensor(df[feat_cols].iloc[idx].values.astype('float32')).unsqueeze(0)

print('=== LIVE HIERARCHICAL ATTENTION DEMO (P001 - P005) ===')
patients = [
    ('P001 (ASD)', 0, 0, 0),
    ('P002 (Control)', 12, 1, 1),
    ('P003 (ASD border)', 1, 5, 2),
    ('P004 (Control)', 13, 2, 3),
    ('P005 (ASD high)', 2, 3, 4)
]

for name, gi, bi, ei in patients:
    g, b, e = get_row(gen_df, gi), get_row(beh_df, bi), get_row(env_df, ei)
    logits, probs, att = model(g, b, e, return_attention=True)
    beta = att['inter_modality'].squeeze().tolist()
    p = probs.item()
    conf = p if p >= 0.5 else 1.0 - p
    pred = 'ASD' if p >= 0.5 else 'Control'
    print(f'{name:<18} -> Pred={pred:<7} P(ASD)={p:.6f} Conf={conf:.6f} beta=[G={beta[0]:.3f}, B={beta[1]:.3f}, E={beta[2]:.3f}] (Sum={sum(beta):.3f})')

print('\n=== LIVE COUNTERFACTUAL EXPLANATION ON P001, P003, P005 ===')
cf_gen = CounterfactualGenerator(model)
for name, gi, bi, ei in [('P001', 0, 0, 0), ('P003', 1, 5, 2), ('P005', 2, 3, 4)]:
    g, b, e = get_row(gen_df, gi), get_row(beh_df, bi), get_row(env_df, ei)
    cf = cf_gen.generate_counterfactual(g, b, e, target_prob=0.35, max_iter=300)
    print(f'{name}:')
    print(f'  Original P(ASD):       {cf["original_probability"]:.4f}')
    print(f'  Target Probability:    <= 0.3500')
    print(f'  Full 32-D Outcome:     {cf["final_probability"]:.4f} (Achieved: {cf["target_achieved"]})')
    print(f'  Top-3 Sparse Outcome:  {cf["top3_sparse_probability"]:.4f}')
    print(f'  L2 Latent Cost:        {cf["l2_cost"]:.4f}')
    print(f'  Optimization Steps:    {cf["iterations"]}')
