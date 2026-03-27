"""
FastAPI Backend for GxE Deep Learning Application.
Provides prediction, metrics, and feature importance endpoints.
"""

import os
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib

from model import GxEModel
from explain import compute_shap_for_prediction

app = FastAPI(
    title="GxE Deep Learning API",
    description="Gene-Environment Interaction model for lung cancer survival prediction",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')

# Global state
model = None
device = None
data = None
metadata = None
scaler_methyl = None


@app.on_event("startup")
async def load_model():
    """Load model and data on startup."""
    global model, device, data, metadata, scaler_methyl

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load metadata
    meta_path = os.path.join(PROCESSED_DIR, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    # Load model config
    config_path = os.path.join(PROCESSED_DIR, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = GxEModel(config['n_genes'], config['n_env'], config['n_methyl']).to(device)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No trained model found. Run train.py first.")
    else:
        print("⚠️ No model config found. Run preprocess.py and train.py first.")

    # Load processed data
    data_path = os.path.join(PROCESSED_DIR, 'processed_data.pkl')
    if os.path.exists(data_path):
        data = joblib.load(data_path)

    # Load scaler
    scaler_path = os.path.join(PROCESSED_DIR, 'scaler_methyl.pkl')
    if os.path.exists(scaler_path):
        scaler_methyl = joblib.load(scaler_path)


# --- Request/Response Models ---

class PatientData(BaseModel):
    age: float
    gender: str  # "MALE" or "FEMALE"
    smoking_history: str  # "never", "former_gt15", "former_le15", "current"
    cancer_stage: str  # "I", "II", "III", "IV"
    gene_values: dict  # {gene_name: value}
    methylation_values: dict  # {cpg_name: value}


class PredictionResponse(BaseModel):
    survival_probability: float
    risk_level: str
    confidence: float
    contributions: list


# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "GxE Deep Learning API", "status": "running"}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """Predict survival probability for a patient."""
    if model is None or data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run preprocessing and training first.")

    try:
        # Encode environment features
        gender_encoded = 1 if patient.gender.upper() == "MALE" else 0
        smoking_map = {"never": 0, "former_gt15": 1, "former_le15": 2, "current": 3}
        smoking_encoded = smoking_map.get(patient.smoking_history, 1)
        stage_map = {"I": 0, "II": 1, "III": 2, "IV": 3}
        stage_encoded = stage_map.get(patient.cancer_stage, 0)

        x_env = np.array([patient.age, gender_encoded, smoking_encoded, stage_encoded], dtype=np.float32)

        # Gene features
        gene_names = data['gene_names']
        x_gene = np.zeros(len(gene_names), dtype=np.float32)
        for i, name in enumerate(gene_names):
            if name in patient.gene_values:
                x_gene[i] = np.log1p(max(0, float(patient.gene_values[name])))

        # Methylation features
        cpg_names = data['cpg_names']
        x_methyl = np.zeros(len(cpg_names), dtype=np.float32)
        for i, name in enumerate(cpg_names):
            if name in patient.methylation_values:
                x_methyl[i] = float(patient.methylation_values[name])

        # Predict
        model.eval()
        with torch.no_grad():
            x_g = torch.FloatTensor(x_gene).unsqueeze(0).to(device)
            x_e = torch.FloatTensor(x_env).unsqueeze(0).to(device)
            x_m = torch.FloatTensor(x_methyl).unsqueeze(0).to(device)
            prob = model(x_g, x_e, x_m).item()

        # Risk level
        if prob < 0.3:
            risk_level = "Low"
        elif prob < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Confidence (distance from 0.5)
        confidence = abs(prob - 0.5) * 2

        # Feature contributions
        contributions, baseline = compute_shap_for_prediction(
            model, x_gene, x_env, x_methyl, data, device
        )

        return PredictionResponse(
            survival_probability=round(float(prob), 4),
            risk_level=risk_level,
            confidence=round(float(confidence), 4),
            contributions=contributions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-metrics")
async def get_model_metrics():
    """Return model evaluation metrics."""
    metrics_path = os.path.join(PROCESSED_DIR, 'model_metrics.json')
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="No metrics found. Train the model first.")
    with open(metrics_path, 'r') as f:
        return json.load(f)


@app.get("/api/feature-importance")
async def get_feature_importance():
    """Return top feature importance scores."""
    fi_path = os.path.join(PROCESSED_DIR, 'feature_importance.json')
    if not os.path.exists(fi_path):
        raise HTTPException(status_code=404, detail="No feature importance data. Run explain.py first.")
    with open(fi_path, 'r') as f:
        return json.load(f)


@app.get("/api/training-history")
async def get_training_history():
    """Return training loss and accuracy curves."""
    hist_path = os.path.join(PROCESSED_DIR, 'training_history.json')
    if not os.path.exists(hist_path):
        raise HTTPException(status_code=404, detail="No training history. Train the model first.")
    with open(hist_path, 'r') as f:
        return json.load(f)


@app.get("/api/metadata")
async def get_metadata():
    """Return dataset metadata for the frontend."""
    if metadata is None:
        raise HTTPException(status_code=404, detail="No metadata found.")
    return metadata


@app.get("/api/sample-patient")
async def get_sample_patient():
    """Return a sample patient for the prediction form."""
    if data is None or metadata is None:
        raise HTTPException(status_code=404, detail="No data loaded.")

    # Use median/mode values from training data
    gene_names = metadata.get('gene_names', [])[:10]
    cpg_names = metadata.get('cpg_names', [])[:5]

    # Get sample values from test data
    sample_genes = {}
    for i, name in enumerate(gene_names):
        idx = data['gene_names'].index(name) if name in data['gene_names'] else -1
        if idx >= 0 and idx < data['X_gene_test'].shape[1]:
            sample_genes[name] = round(float(np.expm1(np.median(data['X_gene_test'][:, idx]))), 2)
        else:
            sample_genes[name] = 5.0

    sample_methylation = {}
    for i, name in enumerate(cpg_names):
        idx = data['cpg_names'].index(name) if name in data['cpg_names'] else -1
        if idx >= 0 and idx < data['X_methyl_test'].shape[1]:
            sample_methylation[name] = round(float(np.median(data['X_methyl_test'][:, idx])), 4)
        else:
            sample_methylation[name] = 0.5

    return {
        "age": 65,
        "gender": "MALE",
        "smoking_history": "former_gt15",
        "cancer_stage": "II",
        "gene_values": sample_genes,
        "methylation_values": sample_methylation,
        "gene_names": gene_names,
        "cpg_names": cpg_names,
    }
