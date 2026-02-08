import torch
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn import MultiTaskTCN
from data.preprocessor import HealthDataPreprocessor


# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 10
MODEL_PATH = "multi_disease_tcn.pth"
# ----------------------------------------


def load_model():
    model = MultiTaskTCN(input_size=13).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_test_prediction(test_records):
    """
    test_records: list of dicts (patient visits)
    """
    preprocessor = HealthDataPreprocessor()

    # Create dummy data to fit the scaler
    from data.synthetic_generator import SyntheticHealthDataGenerator
    generator = SyntheticHealthDataGenerator()
    patients_df, health_df = generator.generate_dataset(n_patients=100)
    
    # Fit the preprocessor with dummy data
    _, _, _ = preprocessor.create_sequences(health_df, patients_df)

    # Convert test data to model-ready tensor
    sequence = preprocessor.prepare_sequence_from_records(
        test_records,
        sequence_length=SEQ_LEN
    ).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        preds = model(sequence)

    print("\n🔍 Model Prediction Results:")
    print(f"Diabetes Risk       : {preds['diabetes'].item():.3f}")
    print(f"Heart Disease Risk  : {preds['heart_disease'].item():.3f}")
    print(f"Kidney Disease Risk : {preds['kidney_disease'].item():.3f}")


if __name__ == "__main__":

    # -------- SAMPLE TEST DATA (MANUAL INPUT) --------
    test_patient_data = [
        {
            "patient_id": 1,
            "visit_date": "2023-01-01",
            "glucose": 110,
            "hba1c": 5.8,
            "creatinine": 1.0,
            "bun": 18,
            "systolic_bp": 130,
            "diastolic_bp": 85,
            "cholesterol": 210,
            "hdl": 42,
            "ldl": 140,
            "triglycerides": 180,
            "bmi": 27.5,
        },
        {
            "patient_id": 1,
            "visit_date": "2024-01-01",
            "glucose": 145,
            "hba1c": 6.9,
            "creatinine": 1.6,
            "bun": 42,
            "systolic_bp": 155,
            "diastolic_bp": 95,
            "cholesterol": 255,
            "hdl": 35,
            "ldl": 175,
            "triglycerides": 260,
            "bmi": 30.2,
        }
    ]

    run_test_prediction(test_patient_data)
