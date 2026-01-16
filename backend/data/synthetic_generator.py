import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random

class SyntheticHealthDataGenerator:
    """Generate realistic longitudinal healthcare data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Normal ranges for health metrics
        self.normal_ranges = {
            'glucose': (70, 100),      # mg/dL
            'hba1c': (4.0, 5.6),       # %
            'creatinine': (0.6, 1.2),  # mg/dL
            'bun': (7, 20),            # mg/dL
            'systolic_bp': (90, 120),  # mmHg
            'diastolic_bp': (60, 80),  # mmHg
            'cholesterol': (125, 200), # mg/dL
            'hdl': (40, 60),           # mg/dL
            'ldl': (100, 130),         # mg/dL
            'triglycerides': (50, 150), # mg/dL
            'bmi': (18.5, 24.9)        # kg/m²
        }
    
    def generate_patient_profile(self) -> Dict:
        """Generate a patient's baseline characteristics"""
        age = np.random.randint(30, 80)
        gender = random.choice(['M', 'F'])
        
        # Risk factors influence baseline values
        diabetes_risk = np.random.random()
        heart_risk = np.random.random()
        kidney_risk = np.random.random()
        
        return {
            'age': age,
            'gender': gender,
            'diabetes_risk': diabetes_risk,
            'heart_risk': heart_risk,
            'kidney_risk': kidney_risk
        }
    
    def generate_baseline_values(self, profile: Dict) -> Dict:
        """Generate initial health values based on patient profile"""
        values = {}
        
        for metric, (low, high) in self.normal_ranges.items():
            base_value = np.random.uniform(low, high)
            
            # Adjust based on risk factors
            if metric in ['glucose', 'hba1c'] and profile['diabetes_risk'] > 0.7:
                base_value *= np.random.uniform(1.2, 1.8)  # Higher glucose for diabetes risk
            
            if metric in ['systolic_bp', 'diastolic_bp', 'cholesterol', 'ldl'] and profile['heart_risk'] > 0.7:
                base_value *= np.random.uniform(1.1, 1.5)  # Higher BP/cholesterol for heart risk
            
            if metric in ['creatinine', 'bun'] and profile['kidney_risk'] > 0.7:
                base_value *= np.random.uniform(1.3, 2.0)  # Higher creatinine for kidney risk
            
            values[metric] = base_value
        
        return values
    
    def simulate_disease_progression(self, baseline_values: Dict, profile: Dict, 
                                   months: int) -> List[Dict]:
        """Simulate how health values change over time"""
        records = []
        current_values = baseline_values.copy()
        
        # Generate irregular visit dates (every 1-6 months)
        visit_dates = []
        current_date = datetime.now() - timedelta(days=months*30)
        
        while current_date < datetime.now():
            visit_dates.append(current_date)
            # Irregular intervals: 1-6 months between visits
            days_to_add = np.random.randint(30, 180)
            current_date += timedelta(days=days_to_add)
        
        for i, visit_date in enumerate(visit_dates):
            # Simulate gradual changes over time
            time_factor = i / len(visit_dates)  # 0 to 1 over time
            
            record = {'visit_date': visit_date}
            
            for metric, baseline_value in current_values.items():
                # Add temporal trends based on disease risks
                trend = 0
                
                if metric in ['glucose', 'hba1c'] and profile['diabetes_risk'] > 0.5:
                    trend = time_factor * 0.3 * profile['diabetes_risk']
                
                if metric in ['systolic_bp', 'cholesterol'] and profile['heart_risk'] > 0.5:
                    trend = time_factor * 0.2 * profile['heart_risk']
                
                if metric in ['creatinine', 'bun'] and profile['kidney_risk'] > 0.5:
                    trend = time_factor * 0.4 * profile['kidney_risk']
                
                # Add noise and trend
                noise = np.random.normal(0, 0.05)  # 5% noise
                new_value = baseline_value * (1 + trend + noise)
                
                # Add missing values (10% chance)
                if np.random.random() < 0.1:
                    record[metric] = None
                else:
                    record[metric] = max(0, new_value)  # Ensure positive values
            
            records.append(record)
        
        return records
    
    def generate_labels(self, records: List[Dict], profile: Dict) -> Dict:
        """Generate disease labels based on final health values and risk factors"""
        if not records:
            return {'diabetes': 0, 'heart_disease': 0, 'kidney_disease': 0}
        
        # Use last record for current health status
        last_record = records[-1]
        
        # Diabetes: based on glucose, HbA1c, and risk profile
        diabetes_score = 0
        glucose = last_record.get('glucose') or 0
        hba1c = last_record.get('hba1c') or 0
        if glucose > 126:  # Diabetes threshold
            diabetes_score += 0.4
        if hba1c > 6.5:   # Diabetes threshold
            diabetes_score += 0.4
        diabetes_score += profile['diabetes_risk'] * 0.2
        
        # Heart disease: based on BP, cholesterol, and risk profile
        heart_score = 0
        systolic_bp = last_record.get('systolic_bp') or 0
        cholesterol = last_record.get('cholesterol') or 0
        if systolic_bp > 140:  # Hypertension
            heart_score += 0.3
        if cholesterol > 240:  # High cholesterol
            heart_score += 0.3
        heart_score += profile['heart_risk'] * 0.4
        
        # Kidney disease: based on creatinine, BUN, and risk profile
        kidney_score = 0
        creatinine = last_record.get('creatinine') or 0
        bun = last_record.get('bun') or 0
        if creatinine > 1.5:  # Kidney dysfunction
            kidney_score += 0.4
        if bun > 40:          # High BUN
            kidney_score += 0.3
        kidney_score += profile['kidney_risk'] * 0.3
        
        return {
            'diabetes': min(1.0, diabetes_score),
            'heart_disease': min(1.0, heart_score),
            'kidney_disease': min(1.0, kidney_score)
        }
    
    def generate_dataset(self, n_patients: int = 1000, 
                        min_months: int = 12, max_months: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete synthetic dataset"""
        patients_data = []
        health_records_data = []
        
        for patient_id in range(n_patients):
            # Generate patient profile
            profile = self.generate_patient_profile()
            baseline_values = self.generate_baseline_values(profile)
            
            # Simulate disease progression
            months = np.random.randint(min_months, max_months)
            records = self.simulate_disease_progression(baseline_values, profile, months)
            
            # Generate labels
            labels = self.generate_labels(records, profile)
            
            # Store patient data
            patients_data.append({
                'patient_id': patient_id,
                'age': profile['age'],
                'gender': profile['gender'],
                'diabetes_label': labels['diabetes'],
                'heart_disease_label': labels['heart_disease'],
                'kidney_disease_label': labels['kidney_disease']
            })
            
            # Store health records
            for record in records:
                record_data = {'patient_id': patient_id}
                record_data.update(record)
                health_records_data.append(record_data)
        
        patients_df = pd.DataFrame(patients_data)
        health_records_df = pd.DataFrame(health_records_data)
        
        return patients_df, health_records_df

# Usage example
if __name__ == "__main__":
    generator = SyntheticHealthDataGenerator()
    patients_df, health_records_df = generator.generate_dataset(n_patients=100)
    
    print(f"Generated {len(patients_df)} patients")
    print(f"Generated {len(health_records_df)} health records")
    print("\nSample patient data:")
    print(patients_df.head())
    print("\nSample health records:")
    print(health_records_df.head())