"""
Data Preprocessing Pipeline for GxE Deep Learning Model
Loads, merges, cleans, and splits TCGA-LUAD data.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'processed')


def load_clinical(path):
    """Load and extract relevant clinical features."""
    print("[1/6] Loading clinical data...")
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df = df.rename(columns={'sampleID': 'sample_id'})

    # Keep relevant columns
    keep_cols = [
        'sample_id',
        'age_at_initial_pathologic_diagnosis',
        'gender',
        'tobacco_smoking_history',
        'pathologic_stage',
        'vital_status',
        'days_to_death',
        'days_to_last_followup',
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].copy()

    # Filter to primary tumor samples only (suffix -01)
    df = df[df['sample_id'].str.endswith('-01', na=False)].copy()
    df = df.drop_duplicates(subset='sample_id', keep='first')

    print(f"   Clinical samples (primary tumor): {len(df)}")
    return df


def load_gene_expression(path, top_n=2000):
    """Load gene expression, transpose, select top-variance genes."""
    print("[2/6] Loading gene expression data...")
    df = pd.read_csv(path, sep='\t', index_col=0, low_memory=False)
    # Genes are rows, samples are columns → transpose
    df = df.T
    df.index.name = 'sample_id'
    df = df.reset_index()

    # Convert to numeric
    gene_cols = [c for c in df.columns if c != 'sample_id']
    df[gene_cols] = df[gene_cols].apply(pd.to_numeric, errors='coerce')

    # Select top-variance genes
    variances = df[gene_cols].var().sort_values(ascending=False)
    top_genes = variances.head(top_n).index.tolist()
    df = df[['sample_id'] + top_genes]

    print(f"   Gene expression: {len(df)} samples, {len(top_genes)} genes selected")
    return df, top_genes


def load_methylation(path, top_n=5000):
    """Load methylation data, transpose, select top-variance CpG sites."""
    print("[3/6] Loading methylation data (this may take a while)...")
    # Read in chunks to handle large file
    chunks = []
    chunk_iter = pd.read_csv(path, sep='\t', index_col=0, chunksize=50000, low_memory=False)
    for i, chunk in enumerate(chunk_iter):
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            print(f"   ... loaded {(i+1)*50000} CpG sites")

    df = pd.concat(chunks)
    print(f"   Total CpG sites loaded: {len(df)}")

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Select top-variance CpG sites
    variances = df.var(axis=1).sort_values(ascending=False)
    top_cpgs = variances.head(top_n).index.tolist()
    df = df.loc[top_cpgs]

    # Transpose: samples as rows
    df = df.T
    df.index.name = 'sample_id'
    df = df.reset_index()

    print(f"   Methylation: {len(df)} samples, {len(top_cpgs)} CpG sites selected")
    return df, top_cpgs


def create_survival_label(df):
    """Create binary survival label from vital_status."""
    print("[4/6] Creating survival labels...")
    df = df.copy()
    df['survival_label'] = (df['vital_status'].str.upper() == 'DECEASED').astype(int)
    print(f"   LIVING: {(df['survival_label']==0).sum()}, DECEASED: {(df['survival_label']==1).sum()}")
    return df


def encode_clinical_features(df):
    """Encode and impute clinical features."""
    print("[5/6] Encoding clinical features...")
    df = df.copy()

    # Age: numeric imputation
    df['age'] = pd.to_numeric(df.get('age_at_initial_pathologic_diagnosis'), errors='coerce')
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age)

    # Gender encoding
    df['gender_encoded'] = (df['gender'].str.upper() == 'MALE').astype(int)

    # Smoking history encoding
    smoking_map = {
        'Lifelong Non-smoker': 0,
        'Current reformed smoker for > 15 years': 1,
        'Current reformed smoker for < or = 15 years': 2,
        'Current reformed smoker, duration not specified': 2,
        'Current smoker': 3,
    }
    df['smoking_encoded'] = df.get('tobacco_smoking_history', pd.Series(dtype=str)).map(smoking_map)
    df['smoking_encoded'] = df['smoking_encoded'].fillna(df['smoking_encoded'].mode().iloc[0]
                                                          if not df['smoking_encoded'].mode().empty else 1)

    # Stage encoding
    stage_map = {}
    for s in ['Stage IA', 'Stage IB', 'Stage I']:
        stage_map[s] = 0
    for s in ['Stage IIA', 'Stage IIB', 'Stage II']:
        stage_map[s] = 1
    for s in ['Stage IIIA', 'Stage IIIB', 'Stage III']:
        stage_map[s] = 2
    for s in ['Stage IV']:
        stage_map[s] = 3
    df['stage_encoded'] = df.get('pathologic_stage', pd.Series(dtype=str)).map(stage_map)
    df['stage_encoded'] = df['stage_encoded'].fillna(df['stage_encoded'].mode().iloc[0]
                                                      if not df['stage_encoded'].mode().empty else 0)

    env_features = ['age', 'gender_encoded', 'smoking_encoded', 'stage_encoded']
    print(f"   Environmental features: {env_features}")
    return df, env_features


def preprocess_and_save():
    """Full preprocessing pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    clinical = load_clinical(os.path.join(DATA_DIR, 'clinical.tsv'))
    gene_expr, gene_names = load_gene_expression(os.path.join(DATA_DIR, 'gene_expression.tsv'), top_n=2000)
    methyl, cpg_names = load_methylation(os.path.join(DATA_DIR, 'methylation.tsv'), top_n=5000)

    # Create labels
    clinical = create_survival_label(clinical)

    # Encode clinical
    clinical, env_features = encode_clinical_features(clinical)

    # Merge all on sample_id
    print("[6/6] Merging datasets...")
    merged = clinical.merge(gene_expr, on='sample_id', how='inner')
    merged = merged.merge(methyl, on='sample_id', how='inner')
    print(f"   Merged dataset: {len(merged)} samples")

    if len(merged) == 0:
        raise ValueError("No samples found after merge. Check sample ID formats.")

    # Prepare feature matrices
    gene_cols = gene_names
    cpg_cols = cpg_names

    # Handle missing values in gene expression
    X_gene = merged[gene_cols].fillna(merged[gene_cols].median())
    # Log1p transform gene expression
    X_gene = np.log1p(X_gene.clip(lower=0))

    # Handle missing values in methylation
    X_methyl = merged[cpg_cols].fillna(merged[cpg_cols].median())
    # Min-max scale methylation
    scaler_methyl = MinMaxScaler()
    X_methyl = pd.DataFrame(
        scaler_methyl.fit_transform(X_methyl),
        columns=cpg_cols, index=X_methyl.index
    )

    # Environment features
    X_env = merged[env_features].astype(float)

    # Labels
    y = merged['survival_label'].values

    # Train/val/test split (70/15/15) with stratification
    idx = np.arange(len(merged))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=0.3, stratify=y, random_state=42
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Save processed data
    data = {
        'X_gene_train': X_gene.iloc[idx_train].values,
        'X_gene_val': X_gene.iloc[idx_val].values,
        'X_gene_test': X_gene.iloc[idx_test].values,
        'X_env_train': X_env.iloc[idx_train].values,
        'X_env_val': X_env.iloc[idx_val].values,
        'X_env_test': X_env.iloc[idx_test].values,
        'X_methyl_train': X_methyl.iloc[idx_train].values,
        'X_methyl_val': X_methyl.iloc[idx_val].values,
        'X_methyl_test': X_methyl.iloc[idx_test].values,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'gene_names': gene_names,
        'cpg_names': cpg_names,
        'env_features': env_features,
        'sample_ids': merged['sample_id'].values,
    }
    joblib.dump(data, os.path.join(OUTPUT_DIR, 'processed_data.pkl'))
    joblib.dump(scaler_methyl, os.path.join(OUTPUT_DIR, 'scaler_methyl.pkl'))

    # Save metadata
    metadata = {
        'n_samples': len(merged),
        'n_genes': len(gene_names),
        'n_cpg_sites': len(cpg_names),
        'n_env_features': len(env_features),
        'n_train': len(idx_train),
        'n_val': len(idx_val),
        'n_test': len(idx_test),
        'class_distribution': {
            'living': int((y == 0).sum()),
            'deceased': int((y == 1).sum()),
        },
        'gene_names': gene_names[:20],  # Save top 20 for frontend
        'cpg_names': cpg_names[:10],     # Save top 10 for frontend
        'env_features': env_features,
    }
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Preprocessing complete!")
    print(f"   Total samples: {len(merged)}")
    print(f"   Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    print(f"   Gene features: {len(gene_names)}, CpG features: {len(cpg_names)}")
    print(f"   Files saved to: {OUTPUT_DIR}")

    return data, metadata


if __name__ == '__main__':
    preprocess_and_save()
