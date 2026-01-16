import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset

class HealthDataPreprocessor:
    """Preprocess healthcare data for temporal modeling"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.feature_columns = [
            'glucose', 'hba1c', 'creatinine', 'bun', 'systolic_bp', 
            'diastolic_bp', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'bmi'
        ]
        self.target_columns = ['diabetes_label', 'heart_disease_label', 'kidney_disease_label']
        self.is_fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in health records"""
        df = df.copy()
        
        # Forward fill within each patient (carry last observation forward)
        df = df.groupby('patient_id').apply(
            lambda group: group.ffill()
        ).reset_index(drop=True)
        
        # Backward fill for remaining missing values
        df = df.groupby('patient_id').apply(
            lambda group: group.bfill()
        ).reset_index(drop=True)
        
        # Fill any remaining missing values with median
        for col in self.feature_columns:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        
        # Sort by patient and date
        df = df.sort_values(['patient_id', 'visit_date'])
        
        # Add time since first visit (in days)
        df['days_since_first'] = df.groupby('patient_id')['visit_date'].transform(
            lambda x: (x - x.min()).dt.days
        )
        
        # Add time between visits (in days)
        df['days_since_last'] = df.groupby('patient_id')['visit_date'].diff().dt.days
        df['days_since_last'] = df['days_since_last'].fillna(0)
        
        return df
    
    def create_sequences(self, health_df: pd.DataFrame, patients_df: pd.DataFrame, 
                        sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for temporal modeling"""
        
        # Merge health records with patient labels
        merged_df = health_df.merge(patients_df[['patient_id'] + self.target_columns], 
                                   on='patient_id', how='left')
        
        # Handle missing values and add time features
        merged_df = self.handle_missing_values(merged_df)
        merged_df = self.add_time_features(merged_df)
        
        sequences = []
        targets = []
        patient_ids = []
        
        for patient_id in merged_df['patient_id'].unique():
            patient_data = merged_df[merged_df['patient_id'] == patient_id].copy()
            patient_data = patient_data.sort_values('visit_date')
            
            if len(patient_data) < 2:  # Need at least 2 visits
                continue
            
            # Extract features and targets
            features = patient_data[self.feature_columns + ['days_since_first', 'days_since_last']].values
            target = patient_data[self.target_columns].iloc[-1].values  # Use final labels
            
            # Create sequences of fixed length
            if len(features) >= sequence_length:
                # Use last sequence_length visits
                seq = features[-sequence_length:]
            else:
                # Pad with zeros if not enough visits
                padding = np.zeros((sequence_length - len(features), features.shape[1]))
                seq = np.vstack([padding, features])
            
            sequences.append(seq)
            targets.append(target)
            patient_ids.append(patient_id)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        patient_ids = np.array(patient_ids)
        
        # Fit scaler on training data
        if not self.is_fitted:
            # Reshape for scaler (combine all sequences)
            all_features = sequences.reshape(-1, sequences.shape[-1])
            self.feature_scaler.fit(all_features)
            self.is_fitted = True
        
        # Normalize sequences
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        sequences_normalized = self.feature_scaler.transform(sequences_flat)
        sequences = sequences_normalized.reshape(original_shape)
        
        return sequences, targets, patient_ids
    
    def split_data(self, sequences: np.ndarray, targets: np.ndarray, 
                   patient_ids: np.ndarray, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Dict:
        """Split data into train/validation/test sets"""
        
        n_samples = len(sequences)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            'train': {
                'sequences': sequences[train_idx],
                'targets': targets[train_idx],
                'patient_ids': patient_ids[train_idx]
            },
            'val': {
                'sequences': sequences[val_idx],
                'targets': targets[val_idx],
                'patient_ids': patient_ids[val_idx]
            },
            'test': {
                'sequences': sequences[test_idx],
                'targets': targets[test_idx],
                'patient_ids': patient_ids[test_idx]
            }
        }
    def prepare_sequence_from_records(
        self,
        records: List[Dict],
        sequence_length: int = 10
    ) -> torch.Tensor:
        """
        Convert raw health records (from DB) into a model-ready sequence tensor
        """

        # Convert records to DataFrame
        df = pd.DataFrame(records)

        if 'patient_id' not in df.columns:
            df['patient_id'] = 0  # dummy ID for grouping

        # Ensure visit_date exists
        df['visit_date'] = pd.to_datetime(df['visit_date'])

        # Apply SAME preprocessing as training
        df = self.handle_missing_values(df)
        df = self.add_time_features(df)

        # Select features in CANONICAL ORDER
        feature_cols = self.feature_columns + ['days_since_first', 'days_since_last']
        features = df[feature_cols].values

        # Pad / truncate
        if len(features) >= sequence_length:
            features = features[-sequence_length:]
        else:
            padding = np.zeros((sequence_length - len(features), features.shape[1]))
            features = np.vstack([padding, features])

        # Scale using SAME scaler
        if not self.is_fitted:
            raise RuntimeError("Preprocessor scaler is not fitted yet.")

        features_scaled = self.feature_scaler.transform(features)

        # Return tensor with batch dimension
        return torch.FloatTensor(features_scaled).unsqueeze(0)


class HealthDataset(Dataset):
    """PyTorch Dataset for health sequences"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, patient_ids: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.patient_ids = patient_ids
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'target': self.targets[idx],
            'patient_id': self.patient_ids[idx]
        }

# Usage example
if __name__ == "__main__":
    from synthetic_generator import SyntheticHealthDataGenerator
    
    # Generate synthetic data
    generator = SyntheticHealthDataGenerator()
    patients_df, health_records_df = generator.generate_dataset(n_patients=100)
    
    # Preprocess data
    preprocessor = HealthDataPreprocessor()
    sequences, targets, patient_ids = preprocessor.create_sequences(
        health_records_df, patients_df, sequence_length=10
    )
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Split data
    data_splits = preprocessor.split_data(sequences, targets, patient_ids)
    print(f"Train: {len(data_splits['train']['sequences'])}")
    print(f"Val: {len(data_splits['val']['sequences'])}")
    print(f"Test: {len(data_splits['test']['sequences'])}")