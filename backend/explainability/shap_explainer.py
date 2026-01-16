import torch
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalExplainer:
    """SHAP-based explainability for medical predictions"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.sequence_length = 10
        self.feature_dim = 13
        
        # Feature names for interpretation
        self.feature_names = [
            'Glucose', 'HbA1c', 'Creatinine', 'BUN', 'Systolic BP', 
            'Diastolic BP', 'Cholesterol', 'HDL', 'LDL', 'Triglycerides', 
            'BMI', 'Days Since First Visit', 'Days Since Last Visit'
        ]
        
        # Medical interpretation mappings
        self.feature_interpretations = {
            'Glucose': 'Blood sugar levels',
            'HbA1c': 'Long-term blood sugar control',
            'Creatinine': 'Kidney function marker',
            'BUN': 'Kidney function and protein metabolism',
            'Systolic BP': 'Heart pumping pressure',
            'Diastolic BP': 'Heart resting pressure',
            'Cholesterol': 'Total cholesterol levels',
            'HDL': 'Good cholesterol',
            'LDL': 'Bad cholesterol',
            'Triglycerides': 'Blood fat levels',
            'BMI': 'Body weight status',
            'Days Since First Visit': 'Disease progression timeline',
            'Days Since Last Visit': 'Visit frequency pattern'
        }
        
        self.disease_names = ['diabetes', 'heart_disease', 'kidney_disease']
    
    def model_predict(self, x):
        """Wrapper function for SHAP compatibility"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            # Return predictions as numpy array
            predictions = np.column_stack([
                outputs['diabetes'].cpu().numpy(),
                outputs['heart_disease'].cpu().numpy(),
                outputs['kidney_disease'].cpu().numpy()
            ])
        return predictions
    
    def predict_disease(self, x, disease_index: int):
        """
        SHAP-compatible prediction function for a single disease.
        x shape: (n_samples, seq_len * feature_dim)
        """
        n_samples = x.shape[0]
        seq_len = self.sequence_length
        feature_dim = self.feature_dim

        # Reshape back to sequence
        x_seq = x.reshape(n_samples, seq_len, feature_dim)
        x_tensor = torch.FloatTensor(x_seq).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_tensor)

        disease_keys = ['diabetes', 'heart_disease', 'kidney_disease']
        disease = disease_keys[disease_index]

        return outputs[disease].cpu().numpy()

    
    def create_explainer(self, background_data: np.ndarray, max_evals: int = 100):
        """Create SHAP explainer with background data"""
        # Use a subset of background data for efficiency
        if len(background_data) > max_evals:
            indices = np.random.choice(len(background_data), max_evals, replace=False)
            background_sample = background_data[indices]
        else:
            background_sample = background_data
        
        # Create SHAP explainer
        self.explainer = shap.Explainer(self.model_predict, background_sample)
        print(f"SHAP explainer created with {len(background_sample)} background samples")
    
    def explain_prediction(self, patient_sequence: np.ndarray, 
                          patient_id: Optional[str] = None) -> Dict:
        """Generate SHAP explanations for a single patient"""
        if not hasattr(self, 'explainer'):
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Get SHAP values
        shap_values = self.explainer(patient_sequence.reshape(1, -1))
        
        # Get model predictions
        predictions = self.model_predict(patient_sequence.reshape(1, -1))[0]
        
        # Reshape SHAP values to match sequence structure
        sequence_length = patient_sequence.shape[0]
        feature_count = patient_sequence.shape[1]
        
        explanations = {}
        
        for i, disease in enumerate(self.disease_names):
            disease_shap = shap_values.values[0, :, i].reshape(sequence_length, feature_count)
            
            explanations[disease] = {
                'prediction': float(predictions[i]),
                'risk_level': self._get_risk_level(predictions[i]),
                'shap_values': disease_shap,
                'feature_importance': self._get_feature_importance(disease_shap),
                'temporal_importance': self._get_temporal_importance(disease_shap),
                'explanation_text': self._generate_explanation_text(
                    disease, predictions[i], disease_shap, patient_sequence
                )
            }
        
        return explanations
    
    def create_disease_explainer(self, background_sequences, disease_index: int):
        """
        Create SHAP KernelExplainer for one disease
        """
        # Flatten sequences
        background_flat = background_sequences.reshape(
            background_sequences.shape[0],
            -1
        )

        def f(x):
            return self.predict_disease(x, disease_index)

        explainer = shap.KernelExplainer(f, background_flat)
        return explainer

    
    def explain_patient(self, patient_sequence, disease_index: int):
        """
        Explain a single patient's disease risk
        """
        # Flatten patient sequence
        patient_flat = patient_sequence.reshape(1, -1)

        explainer = self.create_disease_explainer(
            self.background_data,
            disease_index
        )

        shap_values = explainer.shap_values(patient_flat, nsamples=100)

        # Reshape back to (time, features)
        shap_values = shap_values.reshape(
            self.sequence_length,
            self.feature_dim
        )

        return shap_values

    
    def _get_risk_level(self, prediction: float) -> str:
        """Convert prediction probability to risk level"""
        if prediction < 0.3:
            return "Low Risk"
        elif prediction < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def _get_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance across all time steps"""
        # Sum absolute SHAP values across time for each feature
        feature_importance = np.abs(shap_values).sum(axis=0)
        
        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            importance_dict[feature_name] = float(feature_importance[i])
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        return sorted_importance
    
    def _get_temporal_importance(self, shap_values: np.ndarray) -> List[float]:
        """Calculate importance of each time step"""
        # Sum absolute SHAP values across features for each time step
        temporal_importance = np.abs(shap_values).sum(axis=1)
        return temporal_importance.tolist()
    
    def _generate_explanation_text(self, disease: str, prediction: float, 
                                 shap_values: np.ndarray, patient_sequence: np.ndarray) -> str:
        """Generate human-readable explanation"""
        risk_level = self._get_risk_level(prediction)
        feature_importance = self._get_feature_importance(shap_values)
        
        # Get top 3 most important features
        top_features = list(feature_importance.keys())[:3]
        
        explanation = f"**{disease.replace('_', ' ').title()} Risk: {risk_level}** (Probability: {prediction:.1%})\n\n"
        
        explanation += "**Key Contributing Factors:**\n"
        for i, feature in enumerate(top_features, 1):
            # Get the feature's recent value (last time step)
            feature_idx = self.feature_names.index(feature)
            recent_value = patient_sequence[-1, feature_idx]
            
            # Get SHAP contribution (positive = increases risk, negative = decreases risk)
            contribution = shap_values[:, feature_idx].sum()
            
            impact = "increases" if contribution > 0 else "decreases"
            explanation += f"{i}. **{self.feature_interpretations[feature]}**: "
            explanation += f"Current value ({recent_value:.1f}) {impact} risk\n"
        
        # Add temporal insights
        temporal_importance = self._get_temporal_importance(shap_values)
        most_important_visit = np.argmax(temporal_importance)
        
        explanation += f"\n**Timeline Analysis:**\n"
        explanation += f"Visit #{most_important_visit + 1} shows the strongest predictive signals for this condition.\n"
        
        # Add recommendations based on disease type
        explanation += f"\n**Clinical Considerations:**\n"
        if disease == 'diabetes':
            explanation += "Monitor glucose levels and HbA1c regularly. Consider lifestyle modifications."
        elif disease == 'heart_disease':
            explanation += "Monitor blood pressure and cholesterol. Assess cardiovascular risk factors."
        elif disease == 'kidney_disease':
            explanation += "Monitor kidney function markers. Check for underlying causes."
        
        return explanation
    
    def plot_feature_importance(self, explanations: Dict, save_path: Optional[str] = None):
        """Plot feature importance for all diseases"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, disease in enumerate(self.disease_names):
            if disease in explanations:
                importance = explanations[disease]['feature_importance']
                
                # Get top 8 features
                top_features = list(importance.keys())[:8]
                top_values = [importance[feat] for feat in top_features]
                
                # Create horizontal bar plot
                y_pos = np.arange(len(top_features))
                axes[i].barh(y_pos, top_values, color=plt.cm.viridis(i/3))
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(top_features)
                axes[i].set_xlabel('SHAP Importance')
                axes[i].set_title(f'{disease.replace("_", " ").title()} Risk Factors')
                axes[i].invert_yaxis()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_importance(self, explanations: Dict, save_path: Optional[str] = None):
        """Plot temporal importance across visits"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for i, disease in enumerate(self.disease_names):
            if disease in explanations:
                temporal_imp = explanations[disease]['temporal_importance']
                visits = range(1, len(temporal_imp) + 1)
                
                ax.plot(visits, temporal_imp, marker='o', 
                       label=disease.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Visit Number')
        ax.set_ylabel('Temporal Importance')
        ax.set_title('Importance of Each Visit for Disease Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, patient_sequence: np.ndarray, 
                       patient_info: Dict, save_path: Optional[str] = None) -> str:
        """Generate comprehensive explanation report"""
        explanations = self.explain_prediction(patient_sequence)
        
        report = f"# Medical Risk Assessment Report\n\n"
        report += f"**Patient Information:**\n"
        report += f"- Patient ID: {patient_info.get('patient_id', 'N/A')}\n"
        report += f"- Age: {patient_info.get('age', 'N/A')}\n"
        report += f"- Gender: {patient_info.get('gender', 'N/A')}\n"
        report += f"- Assessment Date: {patient_info.get('date', 'N/A')}\n\n"
        
        report += "## Risk Assessment Summary\n\n"
        
        for disease in self.disease_names:
            if disease in explanations:
                exp = explanations[disease]
                report += f"### {disease.replace('_', ' ').title()}\n"
                report += exp['explanation_text'] + "\n\n"
        
        report += "---\n"
        report += "*This assessment is based on AI analysis and should be reviewed by a qualified healthcare professional.*\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report

# Usage example
if __name__ == "__main__":
    from data.synthetic_generator import SyntheticHealthDataGenerator
    from data.preprocessor import HealthDataPreprocessor
    from models.tcn import MultiTaskTCN
    
    # Generate sample data
    generator = SyntheticHealthDataGenerator()
    patients_df, health_records_df = generator.generate_dataset(n_patients=100)
    
    preprocessor = HealthDataPreprocessor()
    sequences, targets, patient_ids = preprocessor.create_sequences(
        health_records_df, patients_df, sequence_length=10
    )
    
    # Initialize model (assume it's trained)
    model = MultiTaskTCN(input_size=13)
    
    # Create explainer
    explainer = MedicalExplainer(model)
    explainer.create_explainer(sequences[:50])  # Use first 50 as background
    
    # Explain a single patient
    patient_sequence = sequences[0]
    explanations = explainer.explain_prediction(patient_sequence)
    
    print("Explanations generated for patient:")
    for disease, exp in explanations.items():
        print(f"{disease}: {exp['risk_level']} ({exp['prediction']:.1%})")
    
    # Generate report
    patient_info = {'patient_id': 'P001', 'age': 45, 'gender': 'M', 'date': '2024-01-15'}
    report = explainer.generate_report(patient_sequence, patient_info)
    print("\nGenerated report:")
    print(report[:500] + "...")