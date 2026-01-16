# Multi-Disease Temporal Risk Prediction System

A healthcare AI system that predicts future risks of diabetes, heart disease, and kidney disease using temporal deep learning models.

## Architecture
- **Models**: Temporal Convolutional Network (TCN) + Time-Series Transformer
- **Backend**: FastAPI + SQLite
- **Frontend**: React.js
- **Explainability**: SHAP

## Project Structure
```
├── backend/
│   ├── models/           # TCN and Transformer implementations
│   ├── data/            # Data processing and synthetic generation
│   ├── api/             # FastAPI endpoints
│   └── explainability/ # SHAP integration
├── frontend/            # React.js UI
├── data/               # Raw and processed datasets
└── notebooks/          # Development and testing
```

## Setup Instructions
1. Install Python dependencies: `pip install -r requirements.txt`
2. Run backend: `cd backend && python main.py`
3. Run frontend: `cd frontend && npm start`