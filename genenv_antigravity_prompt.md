# 🧬 GxE Deep Learning App — Master Antigravity Prompt
> Copy and paste the prompt below entirely into Antigravity to build your complete application.

---

## 📁 Your Folder Structure
```
genenv/
└── data/
    ├── gene_expression.tsv   # Gene-level RNAseq expression data (IlluminaHiSeq, TCGA-LUAD)
    ├── clinical.tsv          # Patient clinical and environmental data
    └── methylation.tsv       # DNA methylation beta values (HumanMethylation450)
```

---

## 🚀 MASTER PROMPT — Paste This Into Antigravity

```
I have a project folder named genenv. Inside it is a subfolder called data containing three files:
- gene_expression.tsv — gene-level RNAseq expression data (IlluminaHiSeq, TCGA-LUAD)
- clinical.tsv — patient clinical and environmental data (smoking history, age, gender, stage, survival)
- methylation.tsv — DNA methylation beta values (HumanMethylation450)

All files are tab-separated. Sample IDs are shared across all three files and should be used to merge them.

---

TASK: Build a complete full-stack web application for Gene-Environment Interaction (GxE)
modeling using deep learning to predict lung cancer survival outcome.

---

BACKEND (Python + FastAPI):

1. DATA PREPROCESSING:
   - Load and merge all 3 TSV files on sample ID
   - Handle missing values using median imputation for numeric, mode for categorical
   - Encode categorical variables (smoking history, gender, cancer stage)
   - Normalize gene expression using log1p transformation
   - Normalize methylation beta values using min-max scaling
   - Split into train (70%), validation (15%), test (15%) with stratification on survival status

2. DEEP LEARNING MODEL (PyTorch):
   - Build a dual-branch neural network:
     * Branch 1 (Gene Encoder): Input → BatchNorm → Linear(512) → ReLU → Dropout(0.3) → Linear(256) → ReLU
     * Branch 2 (Environment Encoder): Input → BatchNorm → Linear(128) → ReLU → Dropout(0.3) → Linear(64) → ReLU
     * Interaction Layer: Concatenate both branches → Linear(256) → ReLU → Dropout(0.3) → Linear(128) → ReLU
     * Methylation Encoder: Input → Linear(256) → ReLU → Linear(128) → ReLU
     * Final Fusion: Concatenate all 3 encoders → Linear(256) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid
   - Loss function: Binary Cross Entropy for survival prediction
   - Optimizer: Adam with learning rate 0.001 and weight decay 1e-5
   - Learning rate scheduler: ReduceLROnPlateau
   - Early stopping with patience of 10 epochs
   - Train for maximum 100 epochs with batch size 32
   - Save best model as best_model.pth
   - Evaluate using: Accuracy, AUC-ROC, F1-Score, C-index

3. EXPLAINABILITY:
   - Implement SHAP values to show which genes and environmental factors
     contributed most to each prediction
   - Generate feature importance plot

4. FastAPI BACKEND:
   - POST /api/predict — accepts patient gene, methylation, and clinical data as JSON,
     returns survival prediction probability and SHAP explanation
   - GET /api/model-metrics — returns accuracy, AUC, F1, C-index of trained model
   - GET /api/feature-importance — returns top 20 most important features
   - GET /api/training-history — returns loss and accuracy curves per epoch
   - Enable CORS for localhost:3000

---

FRONTEND (React + TypeScript + Tailwind CSS):

1. LANDING PAGE:
   - Hero section explaining GxE modeling and lung cancer prediction
   - Clean modern medical/scientific UI theme (dark blue and white)
   - Navigation: Home, Predict, Model Insights, About

2. PREDICTION PAGE:
   - Input form with:
     * Patient clinical info: Age (slider), Gender (dropdown), Smoking history
       (dropdown: never/former/current), Cancer stage (dropdown: I/II/III/IV)
     * Gene expression input: Top 10 most important genes as number inputs with tooltips
     * Methylation input: Top 5 methylation sites as number inputs
   - Submit button → calls /api/predict
   - Results panel showing:
     * Survival probability as animated circular gauge (0-100%)
     * Risk level: Low / Medium / High with color coding (green/yellow/red)
     * SHAP waterfall chart showing which features pushed prediction up or down
     * Confidence interval display

3. MODEL INSIGHTS PAGE:
   - Training loss and accuracy curves (line charts using Recharts)
   - AUC-ROC curve visualization
   - Feature importance bar chart (top 20 features)
   - Model performance metrics dashboard (Accuracy, AUC, F1, C-index) as cards
   - Gene vs Environment contribution pie chart

4. ABOUT PAGE:
   - Explain the GxE interaction concept
   - Show the neural network architecture as an interactive diagram
   - Dataset description and citations

5. TECH STACK:
   - React 18 + TypeScript
   - Tailwind CSS for styling
   - Recharts for all data visualizations
   - Axios for API calls
   - Framer Motion for animations

---

ADDITIONAL REQUIREMENTS:

- Add a loading spinner while prediction is being calculated
- Add error handling and user-friendly error messages
- Make the app fully responsive (mobile + desktop)
- Add a sample patient button that auto-fills the form with example values
- Store prediction history in localStorage so users can review past predictions
- Add a model confidence indicator showing how certain the model is

---

PROJECT STRUCTURE TO CREATE:

genenv/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── model.py             # PyTorch GxE model
│   ├── preprocess.py        # Data preprocessing pipeline
│   ├── train.py             # Training script
│   ├── explain.py           # SHAP explainability
│   ├── requirements.txt     # All Python dependencies
│   └── best_model.pth       # Saved trained model
├── frontend/
│   ├── src/
│   │   ├── pages/           # Landing, Predict, Insights, About
│   │   ├── components/      # Reusable UI components
│   │   ├── api/             # Axios API calls
│   │   └── App.tsx
│   ├── package.json
│   └── tailwind.config.js
└── data/
    ├── gene_expression.tsv
    ├── clinical.tsv
    └── methylation.tsv

---

START by:
1. First running preprocess.py to load, merge and clean the data
2. Then running train.py to train the model and save best_model.pth
3. Then starting the FastAPI backend with: uvicorn main:app --reload --port 8000
4. Then installing frontend dependencies and starting React with: npm run dev on port 3000
5. Confirm everything is connected and working end to end
```

---

## ✅ Checklist Before Running

- [ ] `gene_expression.tsv` is in `genenv/data/`
- [ ] `clinical.tsv` is in `genenv/data/`
- [ ] `methylation.tsv` is in `genenv/data/`
- [ ] Antigravity is open with `genenv/` as the root folder
- [ ] Paste the full prompt above into Antigravity chat

---

## 🛠️ If Something Breaks, Use This Fix Prompt

```
Something went wrong. Please:
1. Check all file paths are correct relative to the genenv/ root folder
2. Verify all Python dependencies are installed from requirements.txt
3. Make sure FastAPI is running on port 8000 and React on port 3000
4. Check CORS is enabled in main.py for localhost:3000
5. Fix any errors and restart both backend and frontend
```

---

*Project: Deep Learning for Gene–Environment Interaction Modeling*
*Dataset: TCGA-LUAD via UCSC Xena*
*Tool: Antigravity (AI-powered VS Code IDE)*
