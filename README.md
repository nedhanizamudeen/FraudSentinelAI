# FraudSentinel AI v2.2
Autonomous Multi-Agent Financial Fraud Detection System (Inference-Ready Backend)

--------------------------------------------------

OVERVIEW

FraudSentinel is an AI-powered fraud detection system designed to analyze financial transactions using machine learning, rule-based risk scoring, and explainable AI.

This version is inference-ready:
- Models are already trained
- No dataset or training required
- Ready for frontend integration

--------------------------------------------------

CORE FEATURES

- Single transaction fraud prediction
- Batch CSV fraud detection
- Stacking ensemble (XGBoost + LightGBM + CatBoost)
- Inference-time risk adjustment layer
- LLM-based explanations (optional)
- Counterfactual analysis (what to change to avoid fraud)
- Robust handling of noisy, incomplete, and real-world transactional data

--------------------------------------------------

SYSTEM ARCHITECTURE

User Input (JSON / CSV)
        ↓
Input Mapping Layer (Simplified → Internal Feature Space)
        ↓
Feature Pipeline (Pre-trained)
        ↓
Stacking Ensemble Model
        ↓
Adjusted Risk Layer (Rule-based signals)
        ↓
Decision Output
        ↓
Explanation + Counterfactual Analysis

--------------------------------------------------

MODELS USED

- XGBoost
- LightGBM
- CatBoost
- Stacking Meta-Learner (Logistic Regression)

All models are pre-trained and stored in /models/

--------------------------------------------------

DATASET USED

This project was trained using the IEEE-CIS Fraud Detection Dataset, a large-scale real-world financial transaction dataset released as part of the IEEE Computational Intelligence Society Fraud Detection competition on Kaggle.

Dataset characteristics:
- Transaction and identity-based features
- Highly imbalanced fraud classification problem
- Realistic financial transaction patterns
- Designed for advanced fraud detection research

The original training dataset is not included in this repository due to GitHub file size limitations and dataset licensing considerations.

Dataset source:
https://www.kaggle.com/competitions/ieee-fraud-detection

--------------------------------------------------

SETUP

1. Create virtual environment

python -m venv venv

Windows:
venv\Scripts\activate

Linux / Mac:
source venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. (Optional) Add API key for explanations

Create a file named .env and add:

GROQ_API_KEY=your_groq_api_key_here

--------------------------------------------------

USAGE

Single Transaction Prediction

python main.py --predict-single sample_inputs/sample_input.json

Batch CSV Prediction

python main.py --predict-csv sample_inputs/sample_test.csv

--------------------------------------------------

INPUT FORMAT (SIMPLIFIED)

Example JSON:

{
  "amount": 2500,
  "payment_method": "card",
  "card_type": "credit",
  "card_network": "visa",
  "billing_country": "US",
  "shipping_country": "US",
  "email_domain": "gmail.com",
  "receiver_email_domain": "gmail.com"
}

The system automatically maps this to the internal feature space.

--------------------------------------------------

OUTPUT FIELDS

For each transaction:

- FraudProbability
- AdjustedRiskScore
- AdjustedRiskLevel (LOW / MEDIUM / HIGH / CRITICAL)
- FinalVerdict (LEGITIMATE / FRAUD DETECTED / REVIEW)
- PredictedLabel
- EstimatedLoss_USD
- Recommendation
- LLM Explanation (if API key provided)
- Counterfactual Summary

--------------------------------------------------

CSV SUPPORT

The system supports:
- Simple CSVs
- Messy real-world CSVs
- Extra columns (ignored safely)
- Missing columns (auto-filled)
- Alias-based column mapping

--------------------------------------------------

PROJECT STRUCTURE

agents/              -> ML + reasoning agents
tools/               -> utilities
orchestrator/        -> pipeline orchestration
models/              -> trained artifacts
sample_inputs/       -> demo inputs
frontend/            -> UI prototype
main.py              -> entry point
config.py            -> configuration

--------------------------------------------------

FRONTEND

A lightweight frontend prototype is included for local testing and API interaction.

Open in browser:

frontend/index.html

--------------------------------------------------

NOTES

- Training dataset is NOT included
- This package is inference-ready only
- Designed for frontend integration
- Uses pre-trained model artifacts
- Built for experimentation, explainability, and deployment workflows

--------------------------------------------------

PURPOSE

This system demonstrates:

- End-to-end fraud detection pipeline
- Explainable AI in financial systems
- Robust real-world data handling
- Production-style ML system design
- Ensemble learning for fraud classification
- Multi-agent orchestration for modular ML workflows
- API-ready backend architecture

--------------------------------------------------

NEXT STEPS

- Connect frontend to backend
- Deploy as API (Flask / FastAPI)
- Add real-time transaction processing
- Add monitoring and fraud analytics dashboard

--------------------------------------------------

CONNECTED FRONTEND + BACKEND (LOCAL RUN)

1. Install dependencies:

pip install -r requirements.txt

2. Start the connected app:

python app.py

3. Open in browser:

http://127.0.0.1:8000

--------------------------------------------------

API ENDPOINTS

- GET /health
- POST /predict-single
- POST /predict-csv

The API wrapper reuses the existing backend InferenceAgent and does not retrain or modify the original model logic.