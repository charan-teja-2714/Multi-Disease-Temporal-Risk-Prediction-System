# Multi-Disease Temporal Risk Prediction System
## Complete Setup and Running Guide

### 🎯 System Overview
This system predicts future risks of diabetes, heart disease, and kidney disease using:
- **Temporal Convolutional Networks (TCN)** for long-term pattern recognition
- **Time-Series Transformers** for complex temporal relationships
- **Multi-task learning** for joint disease prediction
- **SHAP explainability** for doctor-friendly explanations
- **Professional React UI** for clinical use

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for frontend)
cd frontend
npm install
cd ..
```

### 2. Run Complete Demo
```bash
# This will generate data, train models, and set up the database
python demo.py
```

### 3. Start the System
```bash
# Terminal 1: Start backend API
cd backend
python main.py

# Terminal 2: Start frontend (new terminal)
cd frontend
npm start
```

### 4. Access the System
- **Frontend UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

---

## 📋 Detailed Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 16+
- 8GB RAM (for model training)
- 2GB disk space

### Step-by-Step Setup

#### 1. Python Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Generate Training Data
```bash
# Run data generation script
python -c "
from backend.data.synthetic_generator import SyntheticHealthDataGenerator
generator = SyntheticHealthDataGenerator()
patients_df, health_records_df = generator.generate_dataset(n_patients=1000)
patients_df.to_csv('data/patients.csv', index=False)
health_records_df.to_csv('data/health_records.csv', index=False)
print('Data generated successfully!')
"
```

#### 3. Train AI Models
```bash
# Train models (takes 10-20 minutes)
python -c "
import sys, os
sys.path.append('backend')
from demo import train_models, preprocess_and_split_data
import pandas as pd

patients_df = pd.read_csv('data/patients.csv')
health_records_df = pd.read_csv('data/health_records.csv')
data_splits, _ = preprocess_and_split_data(patients_df, health_records_df)
models, _ = train_models(data_splits)
print('Models trained successfully!')
"
```

#### 4. Setup Database
```bash
# Initialize database with demo data
python -c "
from backend.database import create_tables
create_tables()
print('Database initialized!')
"
```

#### 5. Start Backend API
```bash
cd backend
python main.py
# API will be available at http://localhost:8000
```

#### 6. Start Frontend
```bash
# In a new terminal
cd frontend
npm install  # First time only
npm start
# UI will be available at http://localhost:3000
```

---

## 🧪 Testing the System

### 1. API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Get model information
curl http://localhost:8000/model-info

# Get patients list
curl http://localhost:8000/patients/
```

### 2. Frontend Testing
1. Open http://localhost:3000
2. Navigate to "Patients" → View patient details
3. Add health records via "Add Health Record"
4. Generate predictions by clicking "Generate Prediction"
5. Review AI explanations and risk assessments

### 3. Complete Workflow Test
1. **Add a new patient** (Patients page)
2. **Add health records** (minimum 2 visits required)
3. **Generate prediction** (Patient detail page)
4. **Review explanations** (AI reasoning for each disease)
5. **Monitor trends** (Health charts over time)

---

## 📊 Understanding the AI Models

### Temporal Convolutional Network (TCN)
- **Purpose**: Captures long-term disease progression patterns
- **Architecture**: Dilated causal convolutions with residual connections
- **Strengths**: Fast training, handles long sequences, no vanishing gradients

### Time-Series Transformer
- **Purpose**: Models complex temporal relationships with attention
- **Architecture**: Multi-head self-attention with positional encoding
- **Strengths**: Captures long-range dependencies, handles irregular visits

### Multi-Task Learning
- **Shared Encoder**: Common temporal feature extraction
- **Disease-Specific Heads**: Specialized prediction for each disease
- **Benefits**: Improved accuracy through related disease patterns

---

## 🔍 Explainability Features

### SHAP Integration
- **Feature Importance**: Which health metrics matter most
- **Temporal Importance**: Which visits are most predictive
- **Medical Explanations**: Doctor-friendly reasoning

### Example Explanation Output
```
Diabetes Risk: High Risk (Probability: 78%)

Key Contributing Factors:
1. Blood sugar levels: Current value (145.2) increases risk
2. Long-term blood sugar control: Current value (7.8%) increases risk
3. Body weight status: Current value (28.5) increases risk

Timeline Analysis:
Visit #4 shows the strongest predictive signals for this condition.

Clinical Considerations:
Monitor glucose levels and HbA1c regularly. Consider lifestyle modifications.
```

---

## 🛠️ Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check if port 8000 is in use
netstat -an | findstr :8000

# Try different port
uvicorn main:app --host 0.0.0.0 --port 8001
```

#### Frontend Won't Start
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

#### Model Training Fails
```bash
# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')"

# Reduce batch size if memory is low
# Edit trainer.py: batch_size=16 instead of 32
```

#### Database Issues
```bash
# Reset database
rm medical_predictions.db
python -c "from backend.database import create_tables; create_tables()"
```

### Performance Optimization

#### For Limited Resources
- Reduce dataset size: `n_patients=100` in demo.py
- Use smaller models: `tcn_channels=[16, 16]` in TCN
- Reduce sequence length: `sequence_length=5` in preprocessor

#### For Better Accuracy
- Increase dataset size: `n_patients=5000`
- Use larger models: `d_model=256` in Transformer
- Train longer: `num_epochs=100`

---

## 📁 Project Structure
```
Multi Disease Prediction/
├── backend/
│   ├── models/           # TCN and Transformer implementations
│   ├── data/            # Data processing and generation
│   ├── api/             # FastAPI endpoints
│   ├── explainability/ # SHAP integration
│   └── main.py          # API server
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Main pages
│   │   └── services/    # API communication
│   └── package.json
├── data/               # Generated datasets
├── models/             # Trained model files
├── demo.py            # Complete demo script
└── requirements.txt   # Python dependencies
```

---

## 🎯 Next Steps

### For Development
1. **Add more diseases**: Extend the multi-task framework
2. **Real data integration**: Connect to hospital databases
3. **Advanced explainability**: Add LIME, attention visualizations
4. **Mobile app**: React Native version for tablets

### For Production
1. **Security**: Add authentication, HTTPS, data encryption
2. **Scalability**: Docker containers, load balancing
3. **Monitoring**: Logging, performance metrics, alerts
4. **Compliance**: HIPAA, GDPR data protection

### For Research
1. **Model comparison**: Add GNN, RNN baselines
2. **Ablation studies**: Feature importance analysis
3. **Clinical validation**: Test with real patient data
4. **Publication**: Write research paper on results

---

## 📞 Support

If you encounter issues:
1. Check this README for troubleshooting steps
2. Review the demo.py script for working examples
3. Check API documentation at http://localhost:8000/docs
4. Ensure all dependencies are correctly installed

**System Requirements Met:**
✅ Temporal deep learning models (TCN + Transformer)  
✅ Multi-disease prediction (diabetes, heart, kidney)  
✅ Explainable AI with SHAP  
✅ Professional medical UI  
✅ Complete end-to-end workflow  
✅ Demo-ready with synthetic data  

**Happy predicting! 🏥🤖**