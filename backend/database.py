from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Patient(Base):
    """Patient basic information"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)  # 'M' or 'F'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    health_records = relationship("HealthRecord", back_populates="patient")
    predictions = relationship("Prediction", back_populates="patient")

class HealthRecord(Base):
    """Time-stamped health measurements"""
    __tablename__ = "health_records"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    visit_date = Column(DateTime, nullable=False)
    
    # Lab values (key features for disease prediction)
    glucose = Column(Float)  # mg/dL
    hba1c = Column(Float)    # %
    creatinine = Column(Float)  # mg/dL
    bun = Column(Float)      # mg/dL (blood urea nitrogen)
    systolic_bp = Column(Float)  # mmHg
    diastolic_bp = Column(Float) # mmHg
    cholesterol = Column(Float)  # mg/dL
    hdl = Column(Float)      # mg/dL
    ldl = Column(Float)      # mg/dL
    triglycerides = Column(Float) # mg/dL
    bmi = Column(Float)      # kg/m²
    
    # Relationships
    patient = relationship("Patient", back_populates="health_records")

class Prediction(Base):
    """AI model predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    
    # Risk scores (0-1 probability)
    diabetes_risk = Column(Float, nullable=False)
    heart_disease_risk = Column(Float, nullable=False)
    kidney_disease_risk = Column(Float, nullable=False)
    
    # Explainability (JSON string of SHAP values)
    explanation = Column(String)
    
    # Relationships
    patient = relationship("Patient", back_populates="predictions")

# Database setup
DATABASE_URL = "sqlite:///./medical_predictions.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()