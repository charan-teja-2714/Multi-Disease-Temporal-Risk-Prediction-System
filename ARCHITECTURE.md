# Multi-Disease Temporal Risk Prediction System
## Complete Architecture Documentation

---

## 📋 Table of Contents
1. [System Overview](#1-system-overview)
2. [Architecture Layers](#2-architecture-layers)
3. [Data Flow](#3-data-flow)
4. [Core Components](#4-core-components)
5. [Deep Learning Concepts Explained](#5-deep-learning-concepts-explained)
6. [Current Implementation Status](#6-current-implementation-status)
7. [Setup & Usage](#7-setup--usage)
8. [Technical Decisions](#8-technical-decisions)
9. [Future Extensions](#9-future-extensions)

---

## 1. System Overview

### Purpose
Predict future risks of **diabetes**, **heart disease**, and **kidney disease** by analyzing longitudinal patient health records using temporal deep learning models.

### Key Characteristics
- ✅ **Temporal Analysis**: Handles irregular visit intervals (realistic clinical setting)
- ✅ **Multi-Task Learning**: Predicts 3 diseases jointly with shared representations
- ✅ **Explainable AI**: SHAP-based explanations for medical professionals
- ✅ **Demo-Ready**: Uses synthetic data for demonstration and testing

### Why This Matters
Traditional disease prediction systems analyze single snapshots of patient data. This system understands **how health evolves over time**, capturing trends like:
- Gradual glucose increase → diabetes risk
- Progressive blood pressure elevation → heart disease risk
- Declining kidney function → kidney disease risk

---

## 2. Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                 Frontend (React.js)                      │
│  • Patient Management UI                                 │
│  • Health Record Entry Forms                             │
│  • Risk Visualization Dashboard                          │
│  • Trend Charts & Color-Coded Alerts                     │
└─────────────────────────────────────────────────────────┘
                           ↓ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│              Backend API (FastAPI + SQLite)              │
│  • Patient CRUD Operations                               │
│  • Health Record Management                              │
│  • Prediction Generation                                 │
│  • Database Persistence                                  │
└─────────────────────────────────────────────────────────┘
                           ↓ PyTorch
┌─────────────────────────────────────────────────────────┐
│              AI Models (Deep Learning)                   │
│  • TCN: Temporal Convolutional Network                   │
│  • Transformer: Time-Series Transformer                  │
│  • Multi-Task Heads (3 Disease Predictions)              │
└─────────────────────────────────────────────────────────┘
                           ↓ SHAP
┌─────────────────────────────────────────────────────────┐
│           Explainability Layer (SHAP)                    │
│  • Feature Importance Analysis                           │
│  • Temporal Importance (Which Visits Matter)             │
│  • Doctor-Friendly Medical Explanations                  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Data Storage (SQLite)                       │
│  • Patients Table                                        │
│  • Health Records Table                                  │
│  • Predictions Table                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### Training Phase (Offline)
```
Synthetic Data Generator
         ↓
Generate Realistic Patient Histories
         ↓
Preprocessor (Handle Missing Values, Normalize)
         ↓
Create Fixed-Length Sequences (10 visits)
         ↓
Model Training (TCN/Transformer)
         ↓
Save Trained Model Weights
```

### Inference Phase (Runtime)
```
User Enters Patient Data via UI
         ↓
Data Stored in SQLite Database
         ↓
Retrieve Patient's Health Records
         ↓
Convert to Tensor (10 visits × 13 features)
         ↓
Model Prediction (3 Disease Risks)
         ↓
SHAP Explanation Generation
         ↓
Save Prediction to Database
         ↓
Display Results in UI
```

---

## 4. Core Components

### 4.1 Data Layer

#### **Synthetic Data Generator** (`synthetic_generator.py`)
**Purpose**: Creates realistic longitudinal health data for demonstration.

**What It Does**:
- Generates patient histories spanning 1-5 years
- Simulates disease progression over time
- Creates irregular visit patterns (realistic clinical scenario)
- Adds missing values (10% chance per feature)

**Example Output**:
```
Patient #1:
  Visit 1 (Day 0):   Glucose=95, HbA1c=5.2, BP=120/80
  Visit 2 (Day 45):  Glucose=102, HbA1c=5.4, BP=125/82
  Visit 3 (Day 120): Glucose=115, HbA1c=5.8, BP=130/85
  ...
  → Diabetes Risk: 0.75 (High)
```

#### **Preprocessor** (`preprocessor.py`)
**Purpose**: Cleans and transforms raw data into model-ready format.

**Key Operations**:
1. **Missing Value Handling**:
   - Forward Fill: Use previous visit's value
   - Backward Fill: Use next visit's value
   - Example: If Visit 3 glucose is missing, use Visit 2's value

2. **Normalization** (StandardScaler):
   - Transforms features to mean=0, std=1
   - Why? Models learn better when features are on similar scales
   - Formula: `(value - mean) / std_dev`

3. **Sequence Creation**:
   - Fixed length: 10 visits
   - Padding: If patient has <10 visits, add zeros
   - Truncation: If patient has >10 visits, keep most recent 10

4. **Temporal Features**:
   - `days_since_first`: Days from first visit
   - `days_since_last`: Days from previous visit
   - Helps model understand time gaps

#### **Database Schema** (SQLite)
```sql
-- Patients Table
CREATE TABLE patients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    created_at DATETIME
);

-- Health Records Table
CREATE TABLE health_records (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    visit_date DATE,
    glucose FLOAT,
    hba1c FLOAT,
    creatinine FLOAT,
    bun FLOAT,
    systolic_bp FLOAT,
    diastolic_bp FLOAT,
    cholesterol FLOAT,
    hdl FLOAT,
    ldl FLOAT,
    triglycerides FLOAT,
    bmi FLOAT,
    age FLOAT,
    smoking INTEGER,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

-- Predictions Table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    prediction_date DATETIME,
    diabetes_risk FLOAT,
    heart_disease_risk FLOAT,
    kidney_disease_risk FLOAT,
    explanation TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
```

---

### 4.2 Model Layer

#### **Temporal Convolutional Network (TCN)**

**What is TCN?**
A deep learning architecture designed for sequence modeling using **dilated causal convolutions**.

**Key Concepts**:

1. **Causal Convolution**:
   - Only looks at past and present, never future
   - Essential for time-series (can't predict using future data!)
   ```
   Past → Present → Future
   [✓]    [✓]       [✗]  (Can't use future)
   ```

2. **Dilated Convolution**:
   - Skips elements to capture long-range patterns
   - Dilation = 1: Look at every element
   - Dilation = 2: Skip 1 element
   - Dilation = 4: Skip 3 elements
   ```
   Layer 1 (dilation=1): [●][●][●]
   Layer 2 (dilation=2): [●]_[●]_[●]
   Layer 3 (dilation=4): [●]___[●]___[●]
   ```

3. **Receptive Field**:
   - How far back the model can "see"
   - Grows exponentially with depth
   - Example: 4 layers with dilation [1,2,4,8] → sees 15 time steps back

4. **Residual Connections**:
   - Skip connections that help gradients flow
   - Prevents vanishing gradient problem
   ```
   Input ──────────────────┐
     ↓                     │
   Conv → ReLU → Conv      │
     ↓                     │
   Output ← Add ←──────────┘
   ```

**Architecture**:
```
Input: (batch, 10 visits, 13 features)
         ↓
TCN Block 1 (dilation=1, channels=64)
         ↓
TCN Block 2 (dilation=2, channels=64)
         ↓
TCN Block 3 (dilation=4, channels=64)
         ↓
TCN Block 4 (dilation=8, channels=64)
         ↓
Global Average Pooling
         ↓
Output: (batch, 64 features)
```

**Advantages**:
- ✅ Parallelizable (faster than RNNs)
- ✅ No vanishing gradients
- ✅ Flexible receptive field
- ✅ Stable training

---

#### **Time-Series Transformer**

**What is Transformer?**
An architecture based on **self-attention** that learns which parts of the sequence are important.

**Key Concepts**:

1. **Self-Attention Mechanism**:
   - Compares every visit with every other visit
   - Learns relationships: "Visit 5 is similar to Visit 2"
   - Formula: `Attention(Q,K,V) = softmax(QK^T/√d)V`
   ```
   Visit 1: "How relevant is this to the prediction?"
   Visit 2: "How relevant is this to the prediction?"
   ...
   Visit 10: "How relevant is this to the prediction?"
   
   → Model learns: "Visit 7 is most important!"
   ```

2. **Multi-Head Attention**:
   - Multiple attention mechanisms in parallel
   - Each "head" learns different patterns
   - Example: Head 1 focuses on glucose, Head 2 on blood pressure
   ```
   Input → [Head 1] → Glucose patterns
        → [Head 2] → BP patterns
        → [Head 3] → Kidney markers
        → [Head 4] → Temporal trends
   ```

3. **Positional Encoding**:
   - Adds information about visit order
   - Uses sine/cosine functions
   - Helps model understand "Visit 1 came before Visit 2"

4. **Time Encoding** (Custom for Irregular Visits):
   - Encodes actual time gaps between visits
   - Example: 30 days vs 180 days between visits
   - Formula: `time_encoding = sin(days_gap / 10000^(2i/d))`

**Architecture**:
```
Input: (batch, 10 visits, 13 features)
         ↓
Input Embedding (13 → 64 dimensions)
         ↓
Positional Encoding (add visit order info)
         ↓
Time Encoding (add time gap info)
         ↓
Transformer Encoder Layer 1
  • Multi-Head Attention (4 heads)
  • Feed-Forward Network
  • Layer Normalization
         ↓
Transformer Encoder Layer 2
         ↓
Global Average Pooling
         ↓
Output: (batch, 64 features)
```

**Advantages**:
- ✅ Captures long-range dependencies
- ✅ Interpretable (attention weights show important visits)
- ✅ Handles irregular time gaps naturally
- ✅ State-of-the-art performance

---

#### **Multi-Task Learning Architecture**

**What is Multi-Task Learning?**
Training one model to predict multiple related tasks simultaneously.

**Why Use It?**
1. **Medical Correlation**: Diseases are related
   - Diabetes → Kidney Disease (diabetic nephropathy)
   - Diabetes → Heart Disease (cardiovascular complications)
   - High BP → Both heart and kidney disease

2. **Shared Representations**: Common patterns across diseases
   - High glucose affects multiple organs
   - Blood pressure impacts heart and kidneys

3. **Data Efficiency**: Learn from all labels simultaneously

**Architecture**:
```
Input Sequence (10 visits × 13 features)
         ↓
┌────────────────────────────────────┐
│   Shared Temporal Encoder          │
│   (TCN or Transformer)             │
│   Learns common patterns           │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│   Shared Feature Layer             │
│   (128 → 64 neurons)               │
│   Further feature extraction       │
└────────────────────────────────────┘
         ↓
    ┌────┴────┬────────┐
    ↓         ↓        ↓
┌────────┐ ┌────────┐ ┌────────┐
│Diabetes│ │ Heart  │ │ Kidney │
│  Head  │ │  Head  │ │  Head  │
│(64→32) │ │(64→32) │ │(64→32) │
│(32→1)  │ │(32→1)  │ │(32→1)  │
└────────┘ └────────┘ └────────┘
    ↓         ↓        ↓
  Risk      Risk     Risk
 (0-1)     (0-1)    (0-1)
```

**Training**:
```python
# Combined Loss
total_loss = (
    BCE_loss(diabetes_pred, diabetes_true) +
    BCE_loss(heart_pred, heart_true) +
    BCE_loss(kidney_pred, kidney_true)
) / 3
```

---

### 4.3 API Layer (FastAPI)

**Key Endpoints**:

```python
# Patient Management
POST   /patients/              # Create new patient
GET    /patients/              # List all patients
GET    /patients/{id}          # Get patient details
DELETE /patients/{id}          # Delete patient

# Health Records
POST   /health-records/        # Add health record
GET    /health-records/{patient_id}  # Get patient's records

# Predictions
POST   /predict/{patient_id}   # Generate prediction
GET    /predictions/{patient_id}     # Get prediction history
```

**Example Request/Response**:
```json
// POST /predict/123
{
  "patient_id": 123
}

// Response
{
  "diabetes_risk": 0.75,
  "heart_disease_risk": 0.45,
  "kidney_disease_risk": 0.62,
  "explanation": {
    "diabetes": "High glucose (avg: 145) and HbA1c (6.8) indicate risk",
    "heart": "Elevated blood pressure (140/90) is concerning",
    "kidney": "Rising creatinine (1.4) suggests declining function"
  }
}
```

---

### 4.4 Explainability Layer (SHAP)

**What is SHAP?**
**SH**apley **A**dditive ex**P**lanations - A method to explain model predictions.

**Key Concepts**:

1. **Shapley Values** (from Game Theory):
   - Measures each feature's contribution to prediction
   - Fair allocation of "credit" to features
   - Example: "Glucose contributed +0.3 to diabetes risk"

2. **Feature Importance**:
   ```
   Diabetes Prediction = 0.75
   
   Contributions:
   Glucose:      +0.25 (most important)
   HbA1c:        +0.18
   BMI:          +0.12
   Age:          +0.08
   BP:           +0.05
   Others:       +0.07
   ```

3. **Temporal Importance**:
   - Which visits matter most?
   ```
   Visit 1:  0.05 (baseline)
   Visit 5:  0.15 (glucose spike)
   Visit 8:  0.30 (sustained high glucose) ← Most important!
   Visit 10: 0.20 (recent trend)
   ```

4. **Medical Explanations**:
   ```python
   if shap_value['glucose'] > 0.2:
       explanation = "Elevated glucose levels (avg: 145 mg/dL) " \
                     "significantly increase diabetes risk. " \
                     "Normal range: 70-100 mg/dL."
   ```

**Implementation**:
```python
import shap

# Create explainer
explainer = shap.DeepExplainer(model, background_data)

# Get SHAP values
shap_values = explainer.shap_values(patient_data)

# Generate explanation
explanation = generate_medical_explanation(shap_values)
```

---

### 4.5 Frontend Layer (React.js)

**Pages**:

1. **Dashboard** (`/`):
   - System overview
   - Total patients, predictions
   - Recent activity

2. **Patient List** (`/patients`):
   - Browse all patients
   - Create new patient
   - Search and filter

3. **Patient Detail** (`/patients/:id`):
   - View patient history
   - Health trend charts
   - Generate predictions
   - View risk assessment

4. **Add Health Record** (`/patients/:id/add-record`):
   - Form to enter health measurements
   - Input validation
   - Date picker

**Key Features**:
- 📊 Risk visualization (circular progress bars)
- 📈 Health trend charts (line graphs)
- 🎨 Color-coded risk levels:
  - Green: Low risk (0-0.3)
  - Yellow: Medium risk (0.3-0.7)
  - Red: High risk (0.7-1.0)
- 📱 Responsive design (mobile-friendly)

---

## 5. Deep Learning Concepts Explained

### 5.1 Why NOT LSTM/GRU?

**Traditional Approach**: LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Units)

**Problems**:
1. **Sequential Processing**: Must process visits one-by-one (slow)
2. **Vanishing Gradients**: Struggles with long sequences
3. **Limited Parallelization**: Can't use GPU efficiently

**Our Approach**: TCN and Transformer
- ✅ Parallel processing (10x faster training)
- ✅ Better long-term memory
- ✅ More stable training

---

### 5.2 Training Process

**Loss Function**: Binary Cross-Entropy (BCE)
```python
BCE = -[y*log(p) + (1-y)*log(1-p)]

Where:
  y = true label (0 or 1)
  p = predicted probability (0 to 1)
```

**Example**:
```
True Label: 1 (has diabetes)
Prediction: 0.8 (80% probability)
Loss: -[1*log(0.8)] = 0.22 (low loss, good!)

True Label: 0 (no diabetes)
Prediction: 0.8 (80% probability)
Loss: -[1*log(0.2)] = 1.61 (high loss, bad!)
```

**Optimizer**: Adam (Adaptive Moment Estimation)
- Automatically adjusts learning rate
- Combines momentum and RMSprop
- Learning rate: 0.001

**Early Stopping**:
- Monitors validation loss
- Stops if no improvement for 10 epochs
- Prevents overfitting

---

### 5.3 Evaluation Metrics

**AUC-ROC** (Area Under Receiver Operating Characteristic Curve)
- Measures model's ability to distinguish classes
- Range: 0.5 (random) to 1.0 (perfect)
- Interpretation:
  - 0.9-1.0: Excellent
  - 0.8-0.9: Good
  - 0.7-0.8: Fair
  - 0.6-0.7: Poor
  - 0.5-0.6: Fail

**Precision, Recall, F1-Score**:
```
Confusion Matrix:
                Predicted
              Positive  Negative
Actual Pos      TP        FN
       Neg      FP        TN

Precision = TP / (TP + FP)  # Of predicted positives, how many correct?
Recall    = TP / (TP + FN)  # Of actual positives, how many found?
F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
```

---

## 6. Current Implementation Status

### ✅ Fully Implemented
- Synthetic data generation with realistic patterns
- TCN and Transformer model architectures
- Multi-task learning framework
- Basic prediction API endpoints
- React frontend with visualization
- SQLite database persistence
- Patient and health record management

### ⚠️ Partially Implemented
- **Preprocessor**: Has pandas deprecation warnings (functional but noisy)
- **Prediction Pipeline**: Simplified version (bypasses complex preprocessor)
- **Model Training**: Models exist but need training via `demo.py`
- **SHAP Integration**: Implemented in standalone scripts, not in API

### ❌ Not Implemented
- Model persistence/loading in production API
- Full SHAP explainability in API responses
- User authentication/authorization
- Comprehensive input validation
- Error recovery mechanisms
- Model versioning
- A/B testing framework
- Docker deployment

---

## 7. Setup & Usage

### Prerequisites
```bash
Python 3.8+
Node.js 14+
npm or yarn
```

### Installation

**1. Clone Repository**
```bash
cd "Multi Disease Prediction"
```

**2. Install Python Dependencies**
```bash
pip install -r requirements.txt
```

**3. Install Frontend Dependencies**
```bash
cd frontend
npm install
cd ..
```

### Running the System

**Terminal 1: Backend**
```bash
cd backend
python main.py
# Server runs on http://localhost:8000
```

**Terminal 2: Frontend**
```bash
cd frontend
npm start
# UI opens on http://localhost:3000
```

### Usage Workflow

**Step 1: Create Patient**
- Navigate to "Patients" page
- Click "Add Patient"
- Enter name, age, gender

**Step 2: Add Health Records**
- Select patient
- Click "Add Health Record"
- Enter measurements (glucose, BP, etc.)
- Add at least 2 records for prediction

**Step 3: Generate Prediction**
- View patient details
- Click "Generate Prediction"
- View risk scores and explanations

**Step 4: Monitor Trends**
- View health trend charts
- Track risk over time
- Identify concerning patterns

---

## 8. Technical Decisions

### Why TCN?
- **Speed**: 10x faster than LSTM (parallelizable)
- **Stability**: No vanishing gradients
- **Simplicity**: Easier to train and tune
- **Performance**: Matches or beats RNNs on time-series tasks

### Why Transformer?
- **Attention**: Shows which visits are important (interpretability)
- **Flexibility**: Handles irregular time gaps naturally
- **State-of-the-Art**: Best performance on sequence tasks
- **Scalability**: Efficient with modern hardware

### Why Multi-Task Learning?
- **Medical Reality**: Diseases are correlated
- **Efficiency**: One model instead of three
- **Accuracy**: Shared representations improve performance
- **Practicality**: Easier to deploy and maintain

### Why Synthetic Data?
- **Accessibility**: Real medical data requires IRB approval
- **Control**: Can create specific scenarios for testing
- **Privacy**: No patient privacy concerns
- **Demonstration**: Perfect for academic projects

### Why FastAPI?
- **Speed**: Fastest Python web framework
- **Modern**: Async support, type hints
- **Documentation**: Auto-generated API docs
- **Easy**: Simple to learn and use

### Why React?
- **Popularity**: Large ecosystem, many libraries
- **Component-Based**: Reusable UI components
- **Performance**: Virtual DOM for fast updates
- **Developer Experience**: Great tooling and debugging

### Why SQLite?
- **Simplicity**: No server setup required
- **Portability**: Single file database
- **Sufficient**: Adequate for demo/prototype
- **Easy**: Simple to backup and share

---

## 9. Future Extensions

### If Requested (Not Currently Implemented)

**1. Production Deployment**
- Docker containerization
- PostgreSQL for production database
- Redis for caching
- Nginx reverse proxy
- SSL/TLS certificates

**2. Enhanced Features**
- User authentication (JWT tokens)
- Role-based access control (doctor/admin)
- Audit logging
- Data export (PDF reports)
- Email notifications for high-risk patients

**3. Model Improvements**
- Train on real datasets (MIMIC-IV, eICU)
- Hyperparameter tuning
- Ensemble methods
- Model versioning and A/B testing
- Continuous learning pipeline

**4. Additional Diseases**
- Liver disease
- Stroke risk
- Cancer screening
- Mental health indicators

**5. Advanced Analytics**
- Population health analytics
- Risk stratification
- Treatment recommendation
- Cost-effectiveness analysis

**6. Integration**
- EHR system integration (HL7 FHIR)
- Lab system integration
- Pharmacy system integration
- Telemedicine platform integration

---

## 📚 Key Takeaways

### For Viva/Presentation

**1. Novel Approach**:
- Temporal analysis (not just single snapshots)
- Multi-task learning (correlated diseases)
- Explainable AI (doctor-friendly)

**2. Technical Depth**:
- Two state-of-the-art architectures (TCN + Transformer)
- Proper preprocessing pipeline
- Full-stack implementation

**3. Practical Value**:
- Early disease detection
- Personalized risk assessment
- Clinical decision support

**4. Scalability**:
- Modular architecture
- Easy to add new diseases
- Production-ready design

---

## 📖 References

**Deep Learning**:
- TCN: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)
- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)

**Medical AI**:
- MIMIC-IV: "MIMIC-IV, a freely accessible electronic health record dataset" (Johnson et al., 2020)
- Clinical Prediction: "Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record Analysis" (Shickel et al., 2018)

**Explainability**:
- SHAP: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)

---

## 📞 Support

For questions or issues:
1. Check documentation in `/docs` folder
2. Review code comments
3. Consult viva questions in `VIVA_QUESTIONS.md`
4. Test with synthetic data first

---

**Last Updated**: 2024
**Version**: 1.0
**Status**: Academic Project / Prototype
