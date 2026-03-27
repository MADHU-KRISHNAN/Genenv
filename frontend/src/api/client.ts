import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 30000,
});

export interface PatientData {
    age: number;
    gender: string;
    smoking_history: string;
    cancer_stage: string;
    gene_values: Record<string, number>;
    methylation_values: Record<string, number>;
}

export interface PredictionResult {
    survival_probability: number;
    risk_level: string;
    confidence: number;
    contributions: { feature: string; contribution: number }[];
}

export interface ModelMetrics {
    accuracy: number;
    auc_roc: number;
    f1_score: number;
    c_index: number;
}

export interface FeatureImportance {
    top_features: { feature: string; importance: number; category: string }[];
    category_contributions: { gene: number; environment: number; methylation: number };
}

export interface TrainingHistory {
    train_loss: number[];
    val_loss: number[];
    train_acc: number[];
    val_acc: number[];
}

export interface SamplePatient {
    age: number;
    gender: string;
    smoking_history: string;
    cancer_stage: string;
    gene_values: Record<string, number>;
    methylation_values: Record<string, number>;
    gene_names: string[];
    cpg_names: string[];
}

export const predict = (data: PatientData) =>
    api.post<PredictionResult>('/api/predict', data);

export const getMetrics = () =>
    api.get<ModelMetrics>('/api/model-metrics');

export const getFeatureImportance = () =>
    api.get<FeatureImportance>('/api/feature-importance');

export const getTrainingHistory = () =>
    api.get<TrainingHistory>('/api/training-history');

export const getMetadata = () =>
    api.get('/api/metadata');

export const getSamplePatient = () =>
    api.get<SamplePatient>('/api/sample-patient');
