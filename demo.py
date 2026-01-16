"""
Multi-Disease Prediction System - Complete Demo Script

This script demonstrates the full system workflow:
1. Generate synthetic data
2. Train models
3. Test API endpoints
4. Show explainability features

Run this script to see the complete system in action.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our modules
from backend.data.synthetic_generator import SyntheticHealthDataGenerator
from backend.data.preprocessor import HealthDataPreprocessor, HealthDataset
from backend.models.tcn import MultiTaskTCN
from backend.models.transformer import MultiTaskTransformer
from backend.models.trainer import ModelTrainer
from backend.explainability.shap_explainer import MedicalExplainer
from backend.database import create_tables, SessionLocal, Patient, HealthRecord

def generate_demo_data():
    """Generate synthetic healthcare data for demo"""
    print("🔄 Generating synthetic healthcare data...")
    
    generator = SyntheticHealthDataGenerator(seed=42)
    patients_df, health_records_df = generator.generate_dataset(n_patients=500)
    
    print(f"✅ Generated {len(patients_df)} patients with {len(health_records_df)} health records")
    
    # Save data for later use
    patients_df.to_csv('data/demo_patients.csv', index=False)
    health_records_df.to_csv('data/demo_health_records.csv', index=False)
    
    return patients_df, health_records_df

def preprocess_and_split_data(patients_df, health_records_df):
    """Preprocess data and create train/val/test splits"""
    print("🔄 Preprocessing data and creating sequences...")
    
    preprocessor = HealthDataPreprocessor()
    sequences, targets, patient_ids = preprocessor.create_sequences(
        health_records_df, patients_df, sequence_length=10
    )
    
    print(f"✅ Created {len(sequences)} sequences")
    print(f"   - Sequence shape: {sequences.shape}")
    print(f"   - Target shape: {targets.shape}")
    
    # Split data
    data_splits = preprocessor.split_data(sequences, targets, patient_ids)
    
    print(f"   - Train: {len(data_splits['train']['sequences'])} samples")
    print(f"   - Validation: {len(data_splits['val']['sequences'])} samples")
    print(f"   - Test: {len(data_splits['test']['sequences'])} samples")
    
    return data_splits, preprocessor

def train_models(data_splits):
    """Train both TCN and Transformer models"""
    print("🔄 Training AI models...")
    
    # Create datasets and loaders
    train_dataset = HealthDataset(**data_splits['train'])
    val_dataset = HealthDataset(**data_splits['val'])
    test_dataset = HealthDataset(**data_splits['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    models = {}
    
    # Train TCN model
    print("   Training Temporal Convolutional Network (TCN)...")
    tcn_model = MultiTaskTCN(input_size=13, tcn_channels=[32, 32, 32])
    tcn_trainer = ModelTrainer(tcn_model)
    tcn_trainer.train(train_loader, val_loader, num_epochs=20, patience=5, 
                      save_path='models/tcn_model.pth')
    
    # Evaluate TCN
    tcn_results = tcn_trainer.evaluate(test_loader)
    print("   TCN Test Results:")
    for metric, value in tcn_results.items():
        print(f"     {metric}: {value:.3f}")
    
    models['tcn'] = tcn_model
    
    # Train Transformer model
    print("   Training Time-Series Transformer...")
    transformer_model = MultiTaskTransformer(input_size=13, d_model=64, n_heads=4, n_layers=2)
    transformer_trainer = ModelTrainer(transformer_model)
    transformer_trainer.train(train_loader, val_loader, num_epochs=20, patience=5,
                             save_path='models/transformer_model.pth')
    
    # Evaluate Transformer
    transformer_results = transformer_trainer.evaluate(test_loader)
    print("   Transformer Test Results:")
    for metric, value in transformer_results.items():
        print(f"     {metric}: {value:.3f}")
    
    models['transformer'] = transformer_model
    
    # Plot training histories
    tcn_trainer.plot_training_history('plots/tcn_training.png')
    transformer_trainer.plot_training_history('plots/transformer_training.png')
    
    print("✅ Model training completed!")
    return models, test_loader

def demonstrate_explainability(model, test_loader):
    """Demonstrate SHAP explainability features"""
    print("🔄 Demonstrating AI explainability...")
    
    # Get some test data
    test_data = []
    test_targets = []
    for batch in test_loader:
        test_data.append(batch['sequence'])
        test_targets.append(batch['target'])
        if len(test_data) >= 10:  # Get first 10 batches
            break
    
    test_sequences = torch.cat(test_data, dim=0)[:50]  # First 50 samples
    test_labels = torch.cat(test_targets, dim=0)[:50]
    
    # Create explainer
    explainer = MedicalExplainer(model)
    explainer.create_explainer(test_sequences[:20].numpy())  # Use 20 as background
    
    # Explain a few predictions
    for i in range(3):
        print(f"\n   Explaining Patient {i+1}:")
        patient_sequence = test_sequences[i].numpy()
        explanations = explainer.explain_prediction(patient_sequence)
        
        for disease, exp in explanations.items():
            print(f"     {disease.replace('_', ' ').title()}: {exp['risk_level']} "
                  f"({exp['prediction']:.1%})")
        
        # Generate detailed report for first patient
        if i == 0:
            patient_info = {
                'patient_id': f'DEMO_{i+1}',
                'age': 45,
                'gender': 'M',
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            report = explainer.generate_report(patient_sequence, patient_info, 
                                             f'reports/patient_{i+1}_report.md')
            print(f"     📄 Detailed report saved: reports/patient_{i+1}_report.md")
    
    print("✅ Explainability demonstration completed!")

def populate_database(patients_df, health_records_df):
    """Populate database with demo data"""
    print("🔄 Populating database with demo data...")
    
    # Create tables
    create_tables()
    
    db = SessionLocal()
    try:
        # Add patients
        for _, patient_row in patients_df.head(20).iterrows():  # Add first 20 patients
            db_patient = Patient(
                name=f"Demo Patient {patient_row['patient_id']}",
                age=int(patient_row['age']),
                gender=patient_row['gender']
            )
            db.add(db_patient)
        
        db.commit()
        
        # Get patient IDs from database
        db_patients = db.query(Patient).all()
        patient_mapping = {i: db_patient.id for i, db_patient in enumerate(db_patients)}
        
        # Add health records
        for _, record_row in health_records_df.iterrows():
            if record_row['patient_id'] in patient_mapping:
                db_record = HealthRecord(
                    patient_id=patient_mapping[record_row['patient_id']],
                    visit_date=record_row['visit_date'],
                    glucose=record_row.get('glucose'),
                    hba1c=record_row.get('hba1c'),
                    creatinine=record_row.get('creatinine'),
                    bun=record_row.get('bun'),
                    systolic_bp=record_row.get('systolic_bp'),
                    diastolic_bp=record_row.get('diastolic_bp'),
                    cholesterol=record_row.get('cholesterol'),
                    hdl=record_row.get('hdl'),
                    ldl=record_row.get('ldl'),
                    triglycerides=record_row.get('triglycerides'),
                    bmi=record_row.get('bmi')
                )
                db.add(db_record)
        
        db.commit()
        print(f"✅ Added {len(db_patients)} patients and their health records to database")
        
    finally:
        db.close()

def test_api_endpoints():
    """Test API endpoints (requires running backend)"""
    print("🔄 Testing API endpoints...")
    print("   ⚠️  Make sure to start the backend server first:")
    print("      cd backend && python main.py")
    print("   Then test these endpoints:")
    print("      GET  http://localhost:8000/health")
    print("      GET  http://localhost:8000/patients/")
    print("      GET  http://localhost:8000/model-info")
    print("      POST http://localhost:8000/predict/1")

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'plots', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Run complete demo"""
    print("🚀 Multi-Disease Prediction System - Complete Demo")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Step 1: Generate data
    patients_df, health_records_df = generate_demo_data()
    
    # Step 2: Preprocess data
    data_splits, preprocessor = preprocess_and_split_data(patients_df, health_records_df)
    
    # Step 3: Train models
    models, test_loader = train_models(data_splits)
    
    # Step 4: Demonstrate explainability
    demonstrate_explainability(models['tcn'], test_loader)
    
    # Step 5: Populate database
    populate_database(patients_df, health_records_df)
    
    # Step 6: API testing instructions
    test_api_endpoints()
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Start the backend: cd backend && python main.py")
    print("2. Start the frontend: cd frontend && npm start")
    print("3. Open http://localhost:3000 in your browser")
    print("4. Explore the system with the demo data!")

if __name__ == "__main__":
    main()