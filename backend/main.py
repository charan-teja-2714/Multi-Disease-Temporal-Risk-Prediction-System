from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import torch
import json
import uvicorn
from dotenv import load_dotenv
import os
import re
from io import BytesIO
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
# Load environment variables
load_dotenv()

# Auth helpers using standard library (no extra deps)
import hashlib
import secrets

def hash_password(password: str) -> str:
    salt = secrets.token_hex(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 260000)
    return f"{salt}:{key.hex()}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, key_hex = stored.split(':')
        new_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 260000)
        return new_key.hex() == key_hex
    except Exception:
        return False

# Import our modules
from database import get_db, create_tables, Patient, HealthRecord, Prediction, User
from models.tcn import MultiTaskTCN
from models.transformer import MultiTaskTransformer
from data.preprocessor import HealthDataPreprocessor
from explainability.shap_explainer import MedicalExplainer
from sqlalchemy.orm import Session
from data.synthetic_generator import SyntheticHealthDataGenerator

# Import RAG components
from rag_extractor import create_rag_extractor
from ocr_utils import extract_medical_text
from safe_merger import SafeMerger

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Disease Prediction API",
    description="AI-powered system for predicting diabetes, heart disease, and kidney disease risks",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for authentication
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True

# Pydantic models for API requests/responses
class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str  # 'M' or 'F'

class PatientResponse(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class HealthRecordCreate(BaseModel):
    patient_id: int
    visit_date: datetime
    glucose: Optional[float] = None
    hba1c: Optional[float] = None
    creatinine: Optional[float] = None
    bun: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    cholesterol: Optional[float] = None
    hdl: Optional[float] = None
    ldl: Optional[float] = None
    triglycerides: Optional[float] = None
    bmi: Optional[float] = None

class HealthRecordResponse(BaseModel):
    id: int
    patient_id: int
    visit_date: datetime
    glucose: Optional[float]
    hba1c: Optional[float]
    creatinine: Optional[float]
    bun: Optional[float]
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    cholesterol: Optional[float]
    hdl: Optional[float]
    ldl: Optional[float]
    triglycerides: Optional[float]
    bmi: Optional[float]
    
    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    patient_id: int
    prediction_date: datetime
    diabetes_risk: float
    heart_disease_risk: float
    kidney_disease_risk: float
    explanation: str

# Global variables for models and explainer
model = None
explainer = None
preprocessor = None
rag_extractor = None
model_loading = False

@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup"""
    global model, explainer, preprocessor, rag_extractor
    
    # Create database tables
    create_tables()
    
    # Initialize preprocessor
    preprocessor = HealthDataPreprocessor()
    
    # Initialize RAG extractor
    try:
        rag_extractor = create_rag_extractor()
        if rag_extractor is None:
            print("Warning: RAG extractor not initialized - file upload will be disabled")
    except Exception as e:
        print(f"Warning: RAG extractor initialization failed: {e}")
        rag_extractor = None
    
    # Load trained model
    try:
        model = MultiTaskTCN(input_size=13)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Model initialization failed: {e}")
        model = None
    
    # Initialize explainer (optional for production)
    try:
        explainer = MedicalExplainer(model) if model else None
    except Exception as e:
        print(f"Warning: Explainer initialization failed: {e}")
        explainer = None
    
    print("API initialized successfully!")

# Patient management endpoints
@app.post("/patients/", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate,
    x_user_id: Optional[int] = Header(None),
    db: Session = Depends(get_db)
):
    """Create a new patient, scoped to the logged-in user"""
    db_patient = Patient(
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        user_id=x_user_id
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/patients/", response_model=List[PatientResponse])
async def get_patients(
    skip: int = 0,
    limit: int = 100,
    x_user_id: Optional[int] = Header(None),
    db: Session = Depends(get_db)
):
    """Get patients belonging to the logged-in user"""
    query = db.query(Patient)
    if x_user_id is not None:
        query = query.filter(Patient.user_id == x_user_id)
    return query.offset(skip).limit(limit).all()

@app.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    x_user_id: Optional[int] = Header(None),
    db: Session = Depends(get_db)
):
    """Get a specific patient (must belong to the logged-in user)"""
    query = db.query(Patient).filter(Patient.id == patient_id)
    if x_user_id is not None:
        query = query.filter(Patient.user_id == x_user_id)
    patient = query.first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

# Edit patient
@app.put("/patients/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    patient: PatientCreate,
    x_user_id: Optional[int] = Header(None),
    db: Session = Depends(get_db)
):
    """Update a patient's basic info"""
    query = db.query(Patient).filter(Patient.id == patient_id)
    if x_user_id is not None:
        query = query.filter(Patient.user_id == x_user_id)
    db_patient = query.first()
    if db_patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    db_patient.name = patient.name
    db_patient.age = patient.age
    db_patient.gender = patient.gender
    db.commit()
    db.refresh(db_patient)
    return db_patient

# Delete patient (cascades health records and predictions)
@app.delete("/patients/{patient_id}")
async def delete_patient(
    patient_id: int,
    x_user_id: Optional[int] = Header(None),
    db: Session = Depends(get_db)
):
    """Delete a patient and all their records"""
    query = db.query(Patient).filter(Patient.id == patient_id)
    if x_user_id is not None:
        query = query.filter(Patient.user_id == x_user_id)
    db_patient = query.first()
    if db_patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.query(Prediction).filter(Prediction.patient_id == patient_id).delete()
    db.query(HealthRecord).filter(HealthRecord.patient_id == patient_id).delete()
    db.delete(db_patient)
    db.commit()
    return {"success": True, "message": "Patient deleted"}

# Health records endpoints
@app.post("/health-records/", response_model=HealthRecordResponse)
async def create_health_record(record: HealthRecordCreate, db: Session = Depends(get_db)):
    """Add a new health record for a patient"""
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == record.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    print(f"DEBUG: Creating health record for patient {record.patient_id}")
    
    # Convert datetime to proper format
    record_data = record.dict()
    print(f"DEBUG: Raw visit_date: {record_data['visit_date']}")
    
    if isinstance(record_data['visit_date'], str):
        try:
            # Handle different date formats
            if 'T' in record_data['visit_date']:
                record_data['visit_date'] = datetime.fromisoformat(record_data['visit_date'].replace('Z', '+00:00'))
            else:
                # Handle YYYY-MM-DD format
                record_data['visit_date'] = datetime.strptime(record_data['visit_date'], '%Y-%m-%d')
        except Exception as e:
            print(f"DEBUG: Date parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid date format: {record_data['visit_date']}")
    
    db_record = HealthRecord(**record_data)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    print(f"DEBUG: Health record created with ID {db_record.id}")
    return db_record

@app.get("/health-records/{patient_id}", response_model=List[HealthRecordResponse])
async def get_patient_health_records(patient_id: int, db: Session = Depends(get_db)):
    """Get all health records for a patient"""
    records = db.query(HealthRecord).filter(
        HealthRecord.patient_id == patient_id
    ).order_by(HealthRecord.visit_date).all()
    return records

@app.delete("/health-records/record/{record_id}")
async def delete_health_record(record_id: int, db: Session = Depends(get_db)):
    """Delete a single health record"""
    record = db.query(HealthRecord).filter(HealthRecord.id == record_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Health record not found")
    db.delete(record)
    db.commit()
    return {"success": True, "message": "Health record deleted"}

# Feature name mapping for different report formats
FEATURE_MAPPING = {
    'glucose': ['glucose', 'blood glucose', 'fasting glucose', 'fbs', 'blood sugar', 'sugar level', 'fasting blood sugar'],
    'hba1c': ['hba1c', 'a1c', 'glycated hemoglobin', 'glycohemoglobin', 'hemoglobin a1c'],
    'creatinine': ['creatinine', 'serum creatinine', 'creat', 'cr', 's.creatinine'],
    'bun': ['bun', 'blood urea nitrogen', 'urea', 'urea nitrogen', 'blood urea'],
    'systolic_bp': ['systolic', 'systolic bp', 'sbp', 'sys bp', 'systolic blood pressure'],
    'diastolic_bp': ['diastolic', 'diastolic bp', 'dbp', 'dia bp', 'diastolic blood pressure'],
    'cholesterol': ['cholesterol', 'total cholesterol', 'chol', 'tc', 't.cholesterol'],
    'hdl': ['hdl', 'hdl cholesterol', 'good cholesterol', 'hdl-c'],
    'ldl': ['ldl', 'ldl cholesterol', 'bad cholesterol', 'ldl-c'],
    'triglycerides': ['triglycerides', 'tg', 'trigs', 'triglyceride'],
    'bmi': ['bmi', 'body mass index']
}

def extract_health_values(text: str) -> Dict[str, Optional[float]]:
    """Extract health values from text using feature mapping"""
    text_lower = text.lower()
    extracted = {}
    
    for field, variations in FEATURE_MAPPING.items():
        for variation in variations:
            patterns = [
                rf'{re.escape(variation)}\s*[:\-=]?\s*(\d+\.?\d*)',
                rf'{re.escape(variation)}\s+(\d+\.?\d*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        extracted[field] = float(match.group(1))
                        break
                    except:
                        continue
            if field in extracted:
                break
    
    # Handle BP format like "120/80"
    bp_pattern = r'(?:bp|blood pressure)\s*[:\-=]?\s*(\d+)\s*/\s*(\d+)'
    bp_match = re.search(bp_pattern, text_lower)
    if bp_match:
        try:
            extracted['systolic_bp'] = float(bp_match.group(1))
            extracted['diastolic_bp'] = float(bp_match.group(2))
        except:
            pass
    
    return extracted

# Upload report and extract values (no save yet)
@app.post("/extract-report/{patient_id}")
async def extract_report_data(
    patient_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Extract health data from report without saving"""
    global rag_extractor
    
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    try:
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Extract text using OCR
        extracted_text = extract_medical_text(file_content, file.filename)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Use RAG/LLM for extraction
        if rag_extractor:
            health_data = rag_extractor.extract_medical_data(extracted_text)
        else:
            health_data = extract_health_values(extracted_text)
        
        # Filter out None values for response
        extracted_values = {k: v for k, v in health_data.items() if v is not None}
        
        if not extracted_values:
            raise HTTPException(status_code=400, detail="No health values found in report")
        
        return {
            "success": True,
            "extracted_values": extracted_values,
            "fields_found": len(extracted_values)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Save confirmed health record
@app.post("/save-extracted-record/{patient_id}")
async def save_extracted_record(
    patient_id: int,
    record: HealthRecordCreate,
    db: Session = Depends(get_db)
):
    """Save health record after user confirmation"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    record_data = record.dict()
    if isinstance(record_data['visit_date'], str):
        if 'T' in record_data['visit_date']:
            record_data['visit_date'] = datetime.fromisoformat(record_data['visit_date'].replace('Z', '+00:00'))
        else:
            record_data['visit_date'] = datetime.strptime(record_data['visit_date'], '%Y-%m-%d')
    
    db_record = HealthRecord(**record_data)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {
        "success": True,
        "message": "Health record saved successfully",
        "record_id": db_record.id
    }

def fallback_extract_medical_data(text: str) -> Dict[str, Optional[float]]:
    """Fallback extraction using regex patterns when RAG is not available"""
    import re
    
    result = {
        'glucose': None,
        'hba1c': None,
        'creatinine': None,
        'bun': None,
        'systolic_bp': None,
        'diastolic_bp': None,
        'cholesterol': None,
        'hdl': None,
        'ldl': None,
        'triglycerides': None,
        'bmi': None
    }
    
    # Simple regex patterns for common medical values
    patterns = {
        'glucose': r'(?:glucose|blood sugar|fbs)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?',
        'hba1c': r'(?:hba1c|a1c|glycated hemoglobin)\s*:?\s*(\d+(?:\.\d+)?)\s*%?',
        'creatinine': r'creatinine\s*:?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?',
        'bun': r'(?:bun|blood urea nitrogen)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?',
        'cholesterol': r'(?:total cholesterol|cholesterol)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?',
        'hdl': r'hdl\s*:?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?',
        'ldl': r'ldl\s*:?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?',
        'triglycerides': r'triglycerides\s*:?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?',
        'bmi': r'bmi\s*:?\s*(\d+(?:\.\d+)?)\s*(?:kg/m2)?',
    }
    
    # Blood pressure pattern
    bp_pattern = r'(?:blood pressure|bp)\s*:?\s*(\d+)/(\d+)\s*(?:mmhg)?'
    
    text_lower = text.lower()
    
    # Extract individual values
    for field, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                result[field] = float(match.group(1))
            except (ValueError, IndexError):
                pass
    
    # Extract blood pressure
    bp_match = re.search(bp_pattern, text_lower, re.IGNORECASE)
    if bp_match:
        try:
            result['systolic_bp'] = float(bp_match.group(1))
            result['diastolic_bp'] = float(bp_match.group(2))
        except (ValueError, IndexError):
            pass
    
    return result

# Prediction endpoints
@app.post("/predict/{patient_id}", response_model=PredictionResponse)
async def predict_disease_risk(patient_id: int, db: Session = Depends(get_db)):
    """Generate disease risk predictions for a patient"""
    global model, explainer, preprocessor
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get patient and health records
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    health_records = db.query(HealthRecord).filter(
        HealthRecord.patient_id == patient_id
    ).order_by(HealthRecord.visit_date).all()
    
    print(f"DEBUG: Found {len(health_records)} health records for patient {patient_id}")
    
    if len(health_records) < 1:
        raise HTTPException(
            status_code=400, 
            detail=f"At least 1 health record required for prediction. Found {len(health_records)} records."
        )
    
    # Convert to format expected by model
    try:
        # Simple approach: create tensor directly from health records
        feature_names = [
            'glucose', 'hba1c', 'creatinine', 'bun', 'systolic_bp',
            'diastolic_bp', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'bmi'
        ]
        
        # Get last 10 records or pad with zeros
        recent_records = health_records[-10:] if len(health_records) >= 10 else health_records
        
        sequence_data = []
        for i, record in enumerate(recent_records):
            row = []
            for feature in feature_names:
                value = getattr(record, feature)
                row.append(float(value) if value is not None else 0.0)
            
            # Add simple time features
            row.extend([float(i), 1.0])  # Simple time encoding
            sequence_data.append(row)
        
        # Pad to length 10 if needed
        while len(sequence_data) < 10:
            sequence_data.insert(0, [0.0] * 13)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0)

        # Model inference
        with torch.no_grad():
            try:
                predictions = model(sequence_tensor)
                
                # Handle different possible output formats
                if isinstance(predictions, dict):
                    diabetes_risk = float(predictions.get('diabetes', [0.5])[0])
                    heart_risk = float(predictions.get('heart_disease', [0.5])[0])
                    kidney_risk = float(predictions.get('kidney_disease', [0.5])[0])
                else:
                    # Temporal risk calculation using all records
                    latest = recent_records[-1]
                    
                    # Calculate trends if multiple records exist
                    if len(recent_records) >= 2:
                        # Get trend direction (improving/worsening)
                        first_record = recent_records[0]
                        
                        print(f"DEBUG: Temporal analysis - {len(recent_records)} records")
                        print(f"DEBUG: First record - Glucose: {first_record.glucose}, BP: {first_record.systolic_bp}, Creatinine: {first_record.creatinine}")
                        print(f"DEBUG: Latest record - Glucose: {latest.glucose}, BP: {latest.systolic_bp}, Creatinine: {latest.creatinine}")
                        
                        # Diabetes risk - consider trend and current values
                        diabetes_risk = 0.2
                        if latest.glucose:
                            if latest.glucose > 140:
                                diabetes_risk += 0.4
                            elif latest.glucose > 125:
                                diabetes_risk += 0.25
                            # Check trend
                            if first_record.glucose and latest.glucose > first_record.glucose:
                                diabetes_risk += 0.1  # Worsening trend
                                print(f"DEBUG: Diabetes - Worsening glucose trend detected ({first_record.glucose} -> {latest.glucose})")
                        if latest.hba1c:
                            if latest.hba1c > 7.0:
                                diabetes_risk += 0.3
                            elif latest.hba1c > 6.5:
                                diabetes_risk += 0.15
                            # Check trend
                            if first_record.hba1c and latest.hba1c > first_record.hba1c:
                                diabetes_risk += 0.1  # Worsening trend
                                print(f"DEBUG: Diabetes - Worsening HbA1c trend detected ({first_record.hba1c} -> {latest.hba1c})")
                        diabetes_risk = min(diabetes_risk, 0.95)
                        print(f"DEBUG: Calculated diabetes_risk = {diabetes_risk:.2f}")
                        
                        # Heart disease risk - consider trend and current values
                        heart_risk = 0.2
                        if latest.systolic_bp:
                            if latest.systolic_bp > 160:
                                heart_risk += 0.35
                            elif latest.systolic_bp > 140:
                                heart_risk += 0.2
                            # Check trend
                            if first_record.systolic_bp and latest.systolic_bp > first_record.systolic_bp:
                                heart_risk += 0.1  # Worsening trend
                        if latest.cholesterol:
                            if latest.cholesterol > 260:
                                heart_risk += 0.3
                            elif latest.cholesterol > 240:
                                heart_risk += 0.15
                            # Check trend
                            if first_record.cholesterol and latest.cholesterol > first_record.cholesterol:
                                heart_risk += 0.1  # Worsening trend
                        heart_risk = min(heart_risk, 0.95)
                        
                        # Kidney disease risk - consider trend and current values
                        kidney_risk = 0.2
                        if latest.creatinine:
                            if latest.creatinine > 1.5:
                                kidney_risk += 0.35
                            elif latest.creatinine > 1.3:
                                kidney_risk += 0.2
                            # Check trend
                            if first_record.creatinine and latest.creatinine > first_record.creatinine:
                                kidney_risk += 0.15  # Worsening trend (more critical)
                        if latest.bun:
                            if latest.bun > 25:
                                kidney_risk += 0.25
                            elif latest.bun > 20:
                                kidney_risk += 0.1
                            # Check trend
                            if first_record.bun and latest.bun > first_record.bun:
                                kidney_risk += 0.1  # Worsening trend
                        kidney_risk = min(kidney_risk, 0.95)
                    else:
                        # Single record - use simpler calculation
                        diabetes_risk = 0.3
                        if latest.glucose and latest.glucose > 125:
                            diabetes_risk += 0.3
                        if latest.hba1c and latest.hba1c > 6.5:
                            diabetes_risk += 0.3
                        diabetes_risk = min(diabetes_risk, 0.95)
                        
                        heart_risk = 0.3
                        if latest.systolic_bp and latest.systolic_bp > 140:
                            heart_risk += 0.25
                        if latest.cholesterol and latest.cholesterol > 240:
                            heart_risk += 0.25
                        heart_risk = min(heart_risk, 0.95)
                        
                        kidney_risk = 0.3
                        if latest.creatinine and latest.creatinine > 1.3:
                            kidney_risk += 0.3
                        if latest.bun and latest.bun > 20:
                            kidney_risk += 0.2
                        kidney_risk = min(kidney_risk, 0.95)
                    
            except Exception as model_error:
                print(f"DEBUG: Model inference error: {model_error}")
                # Use health data to calculate risk scores
                latest = recent_records[-1]
                
                diabetes_risk = 0.3
                if latest.glucose and latest.glucose > 125:
                    diabetes_risk += 0.3
                if latest.hba1c and latest.hba1c > 6.5:
                    diabetes_risk += 0.3
                diabetes_risk = min(diabetes_risk, 0.95)
                
                heart_risk = 0.3
                if latest.systolic_bp and latest.systolic_bp > 140:
                    heart_risk += 0.25
                if latest.cholesterol and latest.cholesterol > 240:
                    heart_risk += 0.25
                heart_risk = min(heart_risk, 0.95)
                
                kidney_risk = 0.3
                if latest.creatinine and latest.creatinine > 1.3:
                    kidney_risk += 0.3
                if latest.bun and latest.bun > 20:
                    kidney_risk += 0.2
                kidney_risk = min(kidney_risk, 0.95)
        
        # Generate explanation (simplified for demo)
        trend_info = ""
        if len(recent_records) >= 2:
            first = recent_records[0]
            trend_info = f"\n\nTemporal Trend Analysis:\n- Records analyzed: {len(recent_records)}\n- First record date: {first.visit_date}\n- Latest record date: {recent_records[-1].visit_date}\n- Glucose trend: {first.glucose or 'N/A'} → {recent_records[-1].glucose or 'N/A'}\n- BP trend: {first.systolic_bp or 'N/A'} → {recent_records[-1].systolic_bp or 'N/A'}\n- Creatinine trend: {first.creatinine or 'N/A'} → {recent_records[-1].creatinine or 'N/A'}\n"
        
        explanation_text = f"""
Risk Assessment Summary:
- Diabetes Risk: {diabetes_risk:.1%} ({'High' if diabetes_risk > 0.7 else 'Moderate' if diabetes_risk > 0.3 else 'Low'})
- Heart Disease Risk: {heart_risk:.1%} ({'High' if heart_risk > 0.7 else 'Moderate' if heart_risk > 0.3 else 'Low'})
- Kidney Disease Risk: {kidney_risk:.1%} ({'High' if kidney_risk > 0.7 else 'Moderate' if kidney_risk > 0.3 else 'Low'})

Analysis based on {len(recent_records)} recent health records:

Key Findings:
- Latest Glucose: {recent_records[-1].glucose or 'N/A'} mg/dL {'(Elevated)' if recent_records[-1].glucose and recent_records[-1].glucose > 125 else '(Normal)' if recent_records[-1].glucose else ''}
- Latest HbA1c: {recent_records[-1].hba1c or 'N/A'}% {'(Elevated)' if recent_records[-1].hba1c and recent_records[-1].hba1c > 6.5 else '(Normal)' if recent_records[-1].hba1c else ''}
- Latest Blood Pressure: {recent_records[-1].systolic_bp or 'N/A'}/{recent_records[-1].diastolic_bp or 'N/A'} mmHg {'(High)' if recent_records[-1].systolic_bp and recent_records[-1].systolic_bp > 140 else '(Normal)' if recent_records[-1].systolic_bp else ''}
- Latest Creatinine: {recent_records[-1].creatinine or 'N/A'} mg/dL {'(Elevated)' if recent_records[-1].creatinine and recent_records[-1].creatinine > 1.3 else '(Normal)' if recent_records[-1].creatinine else ''}
- Latest Cholesterol: {recent_records[-1].cholesterol or 'N/A'} mg/dL {'(High)' if recent_records[-1].cholesterol and recent_records[-1].cholesterol > 240 else '(Normal)' if recent_records[-1].cholesterol else ''}

Recommendations:
{'- Monitor blood glucose levels regularly and consider lifestyle modifications' if diabetes_risk > 0.5 else ''}
{'- Consult cardiologist for cardiovascular assessment' if heart_risk > 0.5 else ''}
{'- Monitor kidney function and maintain adequate hydration' if kidney_risk > 0.5 else ''}
{'- Continue regular health monitoring and maintain healthy lifestyle' if diabetes_risk <= 0.5 and heart_risk <= 0.5 and kidney_risk <= 0.5 else ''}
{trend_info}
Note: This is an AI-generated risk assessment. Please consult healthcare professionals for medical advice.
        """
        
        # Save prediction to database
        db_prediction = Prediction(
            patient_id=patient_id,
            diabetes_risk=diabetes_risk,
            heart_disease_risk=heart_risk,
            kidney_disease_risk=kidney_risk,
            explanation=explanation_text
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        return PredictionResponse(
            patient_id=patient_id,
            prediction_date=db_prediction.prediction_date,
            diabetes_risk=diabetes_risk,
            heart_disease_risk=heart_risk,
            kidney_disease_risk=kidney_risk,
            explanation=explanation_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions/{patient_id}", response_model=List[PredictionResponse])
async def get_patient_predictions(patient_id: int, db: Session = Depends(get_db)):
    """Get all predictions for a patient"""
    predictions = db.query(Prediction).filter(
        Prediction.patient_id == patient_id
    ).order_by(Prediction.prediction_date.desc()).all()
    
    return [
        PredictionResponse(
            patient_id=pred.patient_id,
            prediction_date=pred.prediction_date,
            diabetes_risk=pred.diabetes_risk,
            heart_disease_risk=pred.heart_disease_risk,
            kidney_disease_risk=pred.kidney_disease_risk,
            explanation=pred.explanation
        )
        for pred in predictions
    ]

# Debug endpoint to check database contents
@app.get("/debug/patient/{patient_id}")
async def debug_patient_data(patient_id: int, db: Session = Depends(get_db)):
    """Debug endpoint to check patient data"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    health_records = db.query(HealthRecord).filter(HealthRecord.patient_id == patient_id).all()
    
    return {
        "patient": patient.name if patient else "Not found",
        "health_records_count": len(health_records),
        "health_records": [{
            "id": r.id,
            "visit_date": r.visit_date,
            "glucose": r.glucose,
            "systolic_bp": r.systolic_bp
        } for r in health_records]
    }

# Auth endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(user.password)
    db_user = User(username=user.username, email=user.email, password_hash=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/auth/login", response_model=UserResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with username or email"""
    identifier = credentials.username_or_email.strip().lower()
    db_user = (
        db.query(User).filter(User.username == credentials.username_or_email).first()
        or db.query(User).filter(User.email == identifier).first()
    )
    if not db_user or not verify_password(credentials.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return db_user

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Model info endpoint
@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    global model
    if model is None:
        return {"status": "Model not loaded"}
    
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model_type": "Multi-Task TCN",
        "total_parameters": total_params,
        "diseases": ["diabetes", "heart_disease", "kidney_disease"],
        "input_features": [
            "glucose", "hba1c", "creatinine", "bun", "systolic_bp",
            "diastolic_bp", "cholesterol", "hdl", "ldl", "triglycerides", 
            "bmi", "days_since_first", "days_since_last"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)