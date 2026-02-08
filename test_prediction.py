#!/usr/bin/env python3
"""
Simple Model Testing Script
Test model predictions without frontend
Usage: python test_prediction.py
"""

import torch
import numpy as np
import sys
import os

# Add backend to path
sys.path.append('backend')

from models.tcn import MultiTaskTCN
from models.transformer import MultiTaskTransformer
from data.synthetic_generator import SyntheticHealthDataGenerator

def create_sample_patient():
    """Create a sample patient with health records"""
    # Sample patient data (10 visits x 13 features)
    sample_data = np.array([
        # Visit 1: Normal values
        [95, 5.2, 0.9, 15, 120, 80, 180, 55, 110, 120, 22.5, 35, 0],
        # Visit 2: Slight increase
        [102, 5.4, 1.0, 16, 125, 82, 185, 52, 115, 125, 23.1, 45, 0],
        # Visit 3: More concerning
        [115, 5.8, 1.1, 18, 130, 85, 190, 50, 120, 130, 24.2, 60, 0],
        # Visit 4: Worsening
        [125, 6.1, 1.2, 20, 135, 88, 195, 48, 125, 135, 25.0, 75, 1],
        # Visit 5: High risk
        [140, 6.5, 1.4, 22, 140, 90, 200, 45, 130, 140, 26.1, 90, 1],
        # Padding for remaining visits (zeros)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    return sample_data

def test_model_prediction(model, patient_data, model_name):
    """Test model prediction on sample patient"""
    model.eval()
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(patient_data).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        # Extract predictions from dictionary
        predictions = torch.stack([
            outputs['diabetes'],
            outputs['heart_disease'],
            outputs['kidney_disease']
        ])
    
    # Convert to probabilities (already sigmoid activated)
    probs = predictions.squeeze().numpy()
    
    return probs

def interpret_risk(risk_score):
    """Interpret risk score"""
    if risk_score < 0.3:
        return "Low", "🟢"
    elif risk_score < 0.7:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

def print_patient_data(patient_data):
    """Print patient health records"""
    features = ['Glucose', 'HbA1c', 'Creatinine', 'BUN', 'Systolic_BP', 
               'Diastolic_BP', 'Cholesterol', 'HDL', 'LDL', 'Triglycerides', 
               'BMI', 'Age', 'Smoking']
    
    print("\n📋 Patient Health Records:")
    print("="*80)
    
    for visit_idx, visit in enumerate(patient_data):
        if np.sum(visit) == 0:  # Skip padded visits
            continue
            
        print(f"\nVisit {visit_idx + 1}:")
        print("-" * 40)
        for i, (feature, value) in enumerate(zip(features, visit)):
            if feature == 'Smoking':
                value_str = "Yes" if value == 1 else "No"
            else:
                value_str = f"{value:.1f}"
            print(f"{feature:<15}: {value_str}")

def main():
    """Main testing function"""
    print("🏥 Multi-Disease Risk Prediction Test")
    print("="*50)
    
    # Create sample patient
    patient_data = create_sample_patient()
    print_patient_data(patient_data)
    
    # Initialize models
    print("\n🤖 Initializing Models...")
    tcn_model = MultiTaskTCN(input_size=13, tcn_channels=[32, 32, 32], kernel_size=3, dropout=0.2)
    transformer_model = MultiTaskTransformer(input_size=13, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    
    print("⚠️  Note: Models are randomly initialized (not trained)")
    
    # Test TCN Model
    print("\n🔍 TCN Model Predictions:")
    print("-" * 30)
    tcn_predictions = test_model_prediction(tcn_model, patient_data, "TCN")
    
    diseases = ['Diabetes', 'Heart Disease', 'Kidney Disease']
    for disease, risk in zip(diseases, tcn_predictions):
        risk_level, emoji = interpret_risk(risk)
        print(f"{disease:<15}: {risk:.3f} ({risk_level}) {emoji}")
    
    # Test Transformer Model
    print("\n🔍 Transformer Model Predictions:")
    print("-" * 35)
    transformer_predictions = test_model_prediction(transformer_model, patient_data, "Transformer")
    
    for disease, risk in zip(diseases, transformer_predictions):
        risk_level, emoji = interpret_risk(risk)
        print(f"{disease:<15}: {risk:.3f} ({risk_level}) {emoji}")
    
    # Compare models
    print("\n📊 Model Comparison:")
    print("-" * 25)
    print(f"{'Disease':<15} {'TCN':<10} {'Transformer':<12} {'Difference':<10}")
    print("-" * 50)
    
    for disease, tcn_risk, trans_risk in zip(diseases, tcn_predictions, transformer_predictions):
        diff = abs(tcn_risk - trans_risk)
        print(f"{disease:<15} {tcn_risk:<10.3f} {trans_risk:<12.3f} {diff:<10.3f}")
    
    print("\n" + "="*50)
    print("✅ Testing Complete!")
    print("\n💡 To get meaningful predictions:")
    print("   1. Train the models using demo.py")
    print("   2. Load trained weights before testing")
    print("   3. Use real patient data")

def interactive_test():
    """Interactive testing mode"""
    print("\n🔧 Interactive Testing Mode")
    print("Enter patient health data (or press Enter for sample):")
    
    # You can add interactive input here if needed
    main()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        main()