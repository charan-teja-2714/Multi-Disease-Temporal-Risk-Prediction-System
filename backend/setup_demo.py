"""
Quick setup script to populate database with demo data
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from database import create_tables, SessionLocal, Patient, HealthRecord
from data.synthetic_generator import SyntheticHealthDataGenerator
from datetime import datetime, timedelta
import random

def setup_demo_data():
    """Create demo patients and health records"""
    print("Setting up demo data...")
    
    # Create tables
    create_tables()
    
    db = SessionLocal()
    try:
        # Check if data already exists
        existing_patients = db.query(Patient).count()
        if existing_patients > 0:
            print(f"Database already has {existing_patients} patients")
            return
        
        # Create 5 demo patients
        patients = [
            {"name": "John Smith", "age": 45, "gender": "M"},
            {"name": "Mary Johnson", "age": 52, "gender": "F"},
            {"name": "Robert Brown", "age": 38, "gender": "M"},
            {"name": "Lisa Davis", "age": 61, "gender": "F"},
            {"name": "Michael Wilson", "age": 49, "gender": "M"}
        ]
        
        for patient_data in patients:
            # Create patient
            patient = Patient(**patient_data)
            db.add(patient)
            db.commit()
            db.refresh(patient)
            
            # Create 3-5 health records for each patient
            num_records = random.randint(3, 5)
            base_date = datetime.now() - timedelta(days=365)
            
            for i in range(num_records):
                # Generate realistic health values
                visit_date = base_date + timedelta(days=i * 90 + random.randint(-30, 30))
                
                record = HealthRecord(
                    patient_id=patient.id,
                    visit_date=visit_date,
                    glucose=random.uniform(80, 140),
                    hba1c=random.uniform(4.5, 8.0),
                    creatinine=random.uniform(0.7, 1.8),
                    bun=random.uniform(10, 35),
                    systolic_bp=random.uniform(110, 160),
                    diastolic_bp=random.uniform(70, 100),
                    cholesterol=random.uniform(150, 280),
                    hdl=random.uniform(35, 70),
                    ldl=random.uniform(80, 180),
                    triglycerides=random.uniform(60, 200),
                    bmi=random.uniform(20, 35)
                )
                db.add(record)
            
            db.commit()
            print(f"Created patient: {patient.name} with {num_records} health records")
        
        print("✅ Demo data setup complete!")
        
    finally:
        db.close()

if __name__ == "__main__":
    setup_demo_data()