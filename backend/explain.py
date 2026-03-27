"""
SHAP Explainability module for the GxE model.
Generates feature importance scores.
"""

import os
import json
import numpy as np
import torch
import joblib

from model import GxEModel

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__))


class ModelWrapper:
    """Wrapper to make the model compatible with SHAP."""

    def __init__(self, model, device, feature_split):
        self.model = model
        self.device = device
        self.feature_split = feature_split  # (n_gene, n_env, n_methyl)

    def predict(self, X):
        """X is a concatenated array of all features."""
        self.model.eval()
        n_gene, n_env, n_methyl = self.feature_split

        X_gene = torch.FloatTensor(X[:, :n_gene]).to(self.device)
        X_env = torch.FloatTensor(X[:, n_gene:n_gene + n_env]).to(self.device)
        X_methyl = torch.FloatTensor(X[:, n_gene + n_env:]).to(self.device)

        with torch.no_grad():
            preds = self.model(X_gene, X_env, X_methyl).cpu().numpy()
        return preds


def compute_feature_importance():
    """Compute feature importance using permutation-based approach."""
    print("Computing feature importance...")

    # Load data and model
    data = joblib.load(os.path.join(PROCESSED_DIR, 'processed_data.pkl'))
    with open(os.path.join(PROCESSED_DIR, 'model_config.json'), 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GxEModel(config['n_genes'], config['n_env'], config['n_methyl']).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth'),
                                      map_location=device, weights_only=True))
    model.eval()

    wrapper = ModelWrapper(model, device, (config['n_genes'], config['n_env'], config['n_methyl']))

    # Use test data
    X_test = np.hstack([data['X_gene_test'], data['X_env_test'], data['X_methyl_test']])

    # Build feature names
    feature_names = (
        data['gene_names'] +
        data['env_features'] +
        data['cpg_names']
    )

    # Permutation importance on a subset
    n_samples = min(50, len(X_test))
    X_sub = X_test[:n_samples]
    baseline_preds = wrapper.predict(X_sub)
    baseline_score = np.mean(baseline_preds)

    importances = {}
    n_features = X_sub.shape[1]

    # For efficiency, compute importance for gene features (aggregated blocks)
    # and individual env/cpg features
    n_gene = config['n_genes']
    n_env = config['n_env']

    print(f"   Computing importance for {n_gene} genes, {n_env} env features, {config['n_methyl']} CpG sites...")

    # Sample a subset of features to test
    # Top gene features by variance
    gene_var = np.var(X_sub[:, :n_gene], axis=0)
    top_gene_idx = np.argsort(gene_var)[-50:]  # Top 50 most variable genes

    # All env features
    env_indices = list(range(n_gene, n_gene + n_env))

    # Top CpG features by variance
    cpg_start = n_gene + n_env
    cpg_var = np.var(X_sub[:, cpg_start:], axis=0)
    top_cpg_idx = np.argsort(cpg_var)[-50:] + cpg_start

    test_indices = list(top_gene_idx) + env_indices + list(top_cpg_idx)

    for idx in test_indices:
        X_permuted = X_sub.copy()
        np.random.shuffle(X_permuted[:, idx])
        perm_preds = wrapper.predict(X_permuted)
        importance = float(np.mean(np.abs(baseline_preds - perm_preds)))
        importances[feature_names[idx]] = importance

    # Sort by importance
    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    top_20 = sorted_imp[:20]

    # Categorize features
    feature_importance = []
    for name, imp in top_20:
        if name in data['gene_names']:
            category = 'gene'
        elif name in data['env_features']:
            category = 'environment'
        else:
            category = 'methylation'
        feature_importance.append({
            'feature': name,
            'importance': round(imp, 6),
            'category': category,
        })

    # Compute category contributions
    gene_imp = sum(v for k, v in importances.items() if k in data['gene_names'])
    env_imp = sum(v for k, v in importances.items() if k in data['env_features'])
    methyl_imp = sum(v for k, v in importances.items() if k in data['cpg_names'])
    total_imp = gene_imp + env_imp + methyl_imp + 1e-10

    category_contributions = {
        'gene': round(gene_imp / total_imp * 100, 1),
        'environment': round(env_imp / total_imp * 100, 1),
        'methylation': round(methyl_imp / total_imp * 100, 1),
    }

    result = {
        'top_features': feature_importance,
        'category_contributions': category_contributions,
    }

    with open(os.path.join(PROCESSED_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Feature importance computed!")
    print(f"   Top 5 features:")
    for fi in feature_importance[:5]:
        print(f"      {fi['feature']} ({fi['category']}): {fi['importance']}")
    print(f"\n   Category contributions: {category_contributions}")

    return result


def compute_shap_for_prediction(model, x_gene, x_env, x_methyl, data, device):
    """Compute simple feature contribution for a single prediction."""
    model.eval()

    # Baseline prediction (all zeros)
    n_gene = x_gene.shape[0]
    n_env = x_env.shape[0]
    n_methyl = x_methyl.shape[0]

    with torch.no_grad():
        pred = model(
            torch.FloatTensor(x_gene).unsqueeze(0).to(device),
            torch.FloatTensor(x_env).unsqueeze(0).to(device),
            torch.FloatTensor(x_methyl).unsqueeze(0).to(device),
        ).item()

        baseline = model(
            torch.zeros(1, n_gene).to(device),
            torch.zeros(1, n_env).to(device),
            torch.zeros(1, n_methyl).to(device),
        ).item()

    # Simple approximation: compute contributions via input perturbation
    contributions = []
    all_features = np.concatenate([x_gene, x_env, x_methyl])
    all_names = data['gene_names'] + data['env_features'] + data['cpg_names']

    # Only compute for env features + top genes
    env_start = n_gene
    check_indices = list(range(n_gene, n_gene + n_env))  # env features
    # Add top 10 most active gene features
    gene_abs = np.abs(x_gene)
    top_gene_idx = np.argsort(gene_abs)[-10:]
    check_indices = list(top_gene_idx) + check_indices

    for idx in check_indices:
        perturbed = all_features.copy()
        perturbed[idx] = 0

        x_g = torch.FloatTensor(perturbed[:n_gene]).unsqueeze(0).to(device)
        x_e = torch.FloatTensor(perturbed[n_gene:n_gene+n_env]).unsqueeze(0).to(device)
        x_m = torch.FloatTensor(perturbed[n_gene+n_env:]).unsqueeze(0).to(device)

        with torch.no_grad():
            perturbed_pred = model(x_g, x_e, x_m).item()

        contribution = pred - perturbed_pred
        contributions.append({
            'feature': all_names[idx],
            'contribution': round(float(contribution), 6),
        })

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return contributions[:15], baseline


if __name__ == '__main__':
    compute_feature_importance()
