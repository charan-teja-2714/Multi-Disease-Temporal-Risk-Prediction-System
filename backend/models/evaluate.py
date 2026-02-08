import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn import MultiTaskTCN
from data.synthetic_generator import SyntheticHealthDataGenerator
from data.preprocessor import HealthDataPreprocessor


# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 10
THRESHOLD = 0.5
MODEL_PATH = "best_model.pth"
# ----------------------------------------


def evaluate():
    # 1. Load trained model
    model = MultiTaskTCN(input_size=13).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Load dataset (same logic as training)
    generator = SyntheticHealthDataGenerator()
    patients_df, health_df = generator.generate_dataset(n_patients=1000)

    preprocessor = HealthDataPreprocessor()
    X, y, patient_ids = preprocessor.create_sequences(
        health_df,
        patients_df,
        sequence_length=SEQ_LEN
    )

    # 3. Train / Test split (same ratio)
    n_test = int(0.2 * len(X))
    X_test = torch.FloatTensor(X[-n_test:]).to(DEVICE)
    y_test = y[-n_test:]

    # 4. Model inference
    with torch.no_grad():
        preds = model(X_test)

    results = {}
    y_true_all = []
    y_prob_all = []

    # 5. Compute metrics per disease
    diseases = ["diabetes", "heart_disease", "kidney_disease"]
    for i, disease in enumerate(diseases):
        y_true_continuous = y_test[:, i]  # Get column i from y_test array
        y_true = (y_true_continuous > THRESHOLD).astype(int)  # Convert to binary
        y_prob = preds[disease].cpu().numpy()

        y_pred = (y_prob >= THRESHOLD).astype(int)
        
        # Store for visualization
        y_true_all.append(y_true)
        y_prob_all.append(y_prob)

        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            print(f"Warning: {disease} has only one class, skipping AUC")
            results[f"{disease}_auc"] = 0.0
        else:
            results[f"{disease}_auc"] = roc_auc_score(y_true, y_prob)
        
        results[f"{disease}_precision"] = precision_score(y_true, y_pred, zero_division=0)
        results[f"{disease}_recall"] = recall_score(y_true, y_pred, zero_division=0)
        results[f"{disease}_f1"] = f1_score(y_true, y_pred, zero_division=0)

    # 6. Print results
    print("\nTest Results:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")
    
    # 7. Generate visualizations
    generate_visualizations(diseases, y_true_all, y_prob_all, results)


def generate_visualizations(diseases, y_true_all, y_prob_all, results):
    """Generate separate ROC, PR curves, and metrics per disease"""

    for i, disease in enumerate(diseases):
        disease_name = disease.replace("_", " ").title()

        # ---------------- ROC CURVE ----------------
        plt.figure(figsize=(6, 5))
        fpr, tpr, _ = roc_curve(y_true_all[i], y_prob_all[i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{disease_name} – ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        roc_path = f'{disease}_roc_curve.png'
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.show()

        # ---------------- PR CURVE ----------------
        plt.figure(figsize=(6, 5))
        precision, recall, _ = precision_recall_curve(
            y_true_all[i], y_prob_all[i]
        )
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'AUC = {pr_auc:.3f}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{disease_name} – Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)

        pr_path = f'{disease}_pr_curve.png'
        plt.tight_layout()
        plt.savefig(pr_path, dpi=300)
        plt.show()

        # ---------------- METRICS BAR CHART ----------------
        plt.figure(figsize=(6, 5))
        metric_names = ['AUC', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            results[f"{disease}_auc"],
            results[f"{disease}_precision"],
            results[f"{disease}_recall"],
            results[f"{disease}_f1"]
        ]

        bars = plt.bar(metric_names, metric_values, alpha=0.8)
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.title(f'{disease_name} – Performance Metrics')
        plt.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                     height + 0.02,
                     f'{height:.3f}',
                     ha='center', fontsize=9)

        metrics_path = f'{disease}_metrics.png'
        plt.tight_layout()
        plt.savefig(metrics_path, dpi=300)
        plt.show()

        print(f"\n📊 {disease_name} visualizations saved:")
        print(f"   - {roc_path}")
        print(f"   - {pr_path}")
        print(f"   - {metrics_path}")



if __name__ == "__main__":
    evaluate()
