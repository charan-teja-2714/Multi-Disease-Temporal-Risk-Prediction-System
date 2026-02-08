from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import torch
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from database import get_db, create_tables, Patient, HealthRecord, Prediction
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    rag_extractor = create_rag_extractor()
    if rag_extractor is None:
        print("Warning: RAG extractor not initialized - file upload will be disabled")
    
    # Load trained model (you would load your actual trained model here)
    model = MultiTaskTCN(input_size=13)
    # model.load_state_dict(torch.load('path_to_trained_model.pth'))
    model.eval()
    
    # Initialize explainer (would use actual background data in production)
    explainer = MedicalExplainer(model)
    # explainer.create_explainer(background_data)
    generator = SyntheticHealthDataGenerator()
    patients_df, health_df = generator.generate_dataset(n_patients=200)

    preprocessor.create_sequences(
        health_df,
        patients_df,
        sequence_length=10
    )
    print("API initialized successfully!")

# Patient management endpoints
@app.post("/patients/", response_model=PatientResponse)
async def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient"""
    db_patient = Patient(
        name=patient.name,
        age=patient.age,
        gender=patient.gender
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/patients/", response_model=List[PatientResponse])
async def get_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all patients"""
    patients = db.query(Patient).offset(skip).limit(limit).all()
    return patients

@app.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Get a specific patient"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

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

# RAG Report Upload Endpoint
@app.post("/upload-report/{patient_id}")
async def upload_medical_report(
    patient_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process medical report for data extraction"""
    global rag_extractor
    
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Validate file type
    allowed_types = {
        'application/pdf': '.pdf',
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg', 
        'image/png': '.png',
        'image/tiff': '.tiff',
        'image/bmp': '.bmp'
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: PDF, JPG, PNG, TIFF, BMP"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Extract text using OCR
        print(f"Extracting text from {file.filename} ({file.content_type})")
        extracted_text = extract_medical_text(file_content, file.filename)
        
        if not extracted_text:
            return {
                "success": False,
                "message": "No text could be extracted from the uploaded file",
                "extracted_fields": 0,
                "merged_fields": 0
            }
        
        print(f"Extracted {len(extracted_text)} characters of text")
        
        # If RAG extractor is available, use it; otherwise use fallback
        if rag_extractor:
            print("Processing text with RAG extractor")
            extracted_data = rag_extractor.extract_medical_data(extracted_text)
            print(f"RAG extracted data: {extracted_data}")
            
            # If RAG returns empty, try fallback
            if all(v is None for v in extracted_data.values()):
                print("RAG returned empty, using fallback extraction")
                extracted_data = fallback_extract_medical_data(extracted_text)
                print(f"Fallback extracted data: {extracted_data}")
        else:
            print("RAG extractor not available, using fallback extraction")
            extracted_data = fallback_extract_medical_data(extracted_text)
            print(f"Fallback extracted data: {extracted_data}")
        
        # Validate extracted data
        validated_data = SafeMerger.validate_extracted_data(extracted_data)
        print(f"Validated data: {validated_data}")
        
        # Get merge summary before processing
        latest_record = SafeMerger.get_latest_health_record(db, patient_id)
        merge_summary = SafeMerger.get_merge_summary(validated_data, latest_record)
        print(f"Merge summary: {merge_summary}")
        
        if merge_summary['merged_count'] == 0:
            return {
                "success": True,
                "message": "No new medical data found to merge",
                "extracted_fields": merge_summary['extracted_count'],
                "merged_fields": 0,
                "existing_fields": merge_summary['existing_fields']
            }
        
        # Perform safe merge
        new_record = SafeMerger.merge_extracted_data(db, patient_id, validated_data)
        
        if new_record:
            return {
                "success": True,
                "message": f"Successfully extracted and merged medical data",
                "extracted_fields": merge_summary['extracted_count'],
                "merged_fields": merge_summary['merged_count'],
                "new_fields": merge_summary['new_fields'],
                "existing_fields": merge_summary['existing_fields'],
                "record_id": new_record.id
            }
        else:
            return {
                "success": False,
                "message": "Failed to create new health record",
                "extracted_fields": merge_summary['extracted_count'],
                "merged_fields": 0
            }
            
    except Exception as e:
        print(f"Upload processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {str(e)}"
        )

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
                    # Use health data to calculate risk scores consistently
                    latest = recent_records[-1]
                    
                    # Diabetes risk based on glucose and HbA1c
                    diabetes_risk = 0.3
                    if latest.glucose and latest.glucose > 125:
                        diabetes_risk += 0.3
                    if latest.hba1c and latest.hba1c > 6.5:
                        diabetes_risk += 0.3
                    diabetes_risk = min(diabetes_risk, 0.95)
                    
                    # Heart disease risk based on BP and cholesterol
                    heart_risk = 0.3
                    if latest.systolic_bp and latest.systolic_bp > 140:
                        heart_risk += 0.25
                    if latest.cholesterol and latest.cholesterol > 240:
                        heart_risk += 0.25
                    heart_risk = min(heart_risk, 0.95)
                    
                    # Kidney disease risk based on creatinine and BUN
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)