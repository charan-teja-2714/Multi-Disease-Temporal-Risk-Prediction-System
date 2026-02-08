import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class MultiTaskLoss(nn.Module):
    """Multi-task loss for joint disease prediction"""
    
    def __init__(self, task_weights: Dict[str, float] = None):
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights or {
            'diabetes': 1.0,
            'heart_disease': 1.0,
            'kidney_disease': 1.0
        }
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        predictions: dict with keys ['diabetes', 'heart_disease', 'kidney_disease']
        targets: tensor of shape (batch_size, 3) with [diabetes, heart_disease, kidney_disease]
        """
        losses = {}
        total_loss = 0
        
        disease_names = ['diabetes', 'heart_disease', 'kidney_disease']
        
        for i, disease in enumerate(disease_names):
            if disease in predictions:
                loss = self.bce_loss(predictions[disease], targets[:, i])
                losses[f'{disease}_loss'] = loss
                total_loss += self.task_weights[disease] * loss
        
        losses['total_loss'] = total_loss
        return losses

class ModelTrainer:
    """Training pipeline for multi-task disease prediction"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = MultiTaskLoss()
        
        # Training history
        self.train_history = {
            'total_loss': [], 'diabetes_loss': [], 'heart_disease_loss': [], 'kidney_disease_loss': []
        }
        self.val_history = {
            'total_loss': [], 'diabetes_loss': [], 'heart_disease_loss': [], 'kidney_disease_loss': [],
            'diabetes_auc': [], 'heart_disease_auc': [], 'kidney_disease_auc': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total_loss': 0, 'diabetes_loss': 0, 'heart_disease_loss': 0, 'kidney_disease_loss': 0}
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            sequences = batch['sequence'].to(self.device)
            targets = batch['target'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Calculate losses
            losses = self.loss_fn(predictions, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            optimizer.step()
            
            # Accumulate losses
            for key, loss in losses.items():
                epoch_losses[key] += loss.item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {'total_loss': 0, 'diabetes_loss': 0, 'heart_disease_loss': 0, 'kidney_disease_loss': 0}
        
        all_predictions = {'diabetes': [], 'heart_disease': [], 'kidney_disease': []}
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Calculate losses
                losses = self.loss_fn(predictions, targets)
                
                # Accumulate losses
                for key, loss in losses.items():
                    epoch_losses[key] += loss.item()
                
                # Store predictions and targets for metrics
                for disease in all_predictions:
                    all_predictions[disease].extend(predictions[disease].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Calculate AUC scores
        all_targets = np.array(all_targets)
        disease_names = ['diabetes', 'heart_disease', 'kidney_disease']
        
        for i, disease in enumerate(disease_names):
            try:
                auc = roc_auc_score(all_targets[:, i], all_predictions[disease])
                epoch_losses[f'{disease}_auc'] = auc
            except ValueError:
                # Handle case where all targets are the same class
                epoch_losses[f'{disease}_auc'] = 0.5
        
        return epoch_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 100, learning_rate: float = 0.001, 
              patience: int = 10, save_path: str = 'multi_disease_tcn.pth'):
        """Full training loop with early stopping"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_losses = self.validate_epoch(val_loader)
            
            # Update learning rate
            scheduler.step(val_losses['total_loss'])
            
            # Store history
            for key, value in train_losses.items():
                self.train_history[key].append(value)
            
            for key, value in val_losses.items():
                self.val_history[key].append(value)
            
            # Print progress
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            print(f"Val AUCs - Diabetes: {val_losses['diabetes_auc']:.3f}, "
                  f"Heart: {val_losses['heart_disease_auc']:.3f}, "
                  f"Kidney: {val_losses['kidney_disease_auc']:.3f}")
            
            # Early stopping
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_losses['total_loss']
                }, save_path)
                print(f"New best model saved with val_loss: {val_losses['total_loss']:.4f}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        
        all_predictions = {'diabetes': [], 'heart_disease': [], 'kidney_disease': []}
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                
                predictions = self.model(sequences)
                
                for disease in all_predictions:
                    all_predictions[disease].extend(predictions[disease].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate comprehensive metrics
        all_targets = np.array(all_targets)
        disease_names = ['diabetes', 'heart_disease', 'kidney_disease']
        results = {}
        
        for i, disease in enumerate(disease_names):
            y_true = all_targets[:, i]
            y_pred_proba = np.array(all_predictions[disease])
            
            # Convert continuous targets to binary (threshold at 0.5)
            y_true_binary = (y_true > 0.5).astype(int)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # AUC
            try:
                auc = roc_auc_score(y_true_binary, y_pred_proba)
            except ValueError:
                auc = 0.5
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_binary, y_pred, average='binary', zero_division=0
            )
            
            results[f'{disease}_auc'] = auc
            results[f'{disease}_precision'] = precision
            results[f'{disease}_recall'] = recall
            results[f'{disease}_f1'] = f1
        
        return results
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_history['total_loss'], label='Train')
        axes[0, 0].plot(self.val_history['total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Individual disease losses
        diseases = ['diabetes', 'heart_disease', 'kidney_disease']
        colors = ['red', 'blue', 'green']
        
        for disease, color in zip(diseases, colors):
            axes[0, 1].plot(self.val_history[f'{disease}_loss'], 
                           label=disease.replace('_', ' ').title(), color=color)
        axes[0, 1].set_title('Validation Losses by Disease')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # AUC curves
        for disease, color in zip(diseases, colors):
            axes[1, 0].plot(self.val_history[f'{disease}_auc'], 
                           label=disease.replace('_', ' ').title(), color=color)
        axes[1, 0].set_title('Validation AUC by Disease')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Final metrics comparison
        final_aucs = [self.val_history[f'{disease}_auc'][-1] for disease in diseases]
        disease_labels = [disease.replace('_', ' ').title() for disease in diseases]
        
        axes[1, 1].bar(disease_labels, final_aucs, color=colors)
        axes[1, 1].set_title('Final Validation AUC')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
if __name__ == "__main__":
    from data.synthetic_generator import SyntheticHealthDataGenerator
    from data.preprocessor import HealthDataPreprocessor, HealthDataset
    from models.tcn import MultiTaskTCN
    
    # Generate and preprocess data
    generator = SyntheticHealthDataGenerator()
    patients_df, health_records_df = generator.generate_dataset(n_patients=1000)
    
    preprocessor = HealthDataPreprocessor()
    sequences, targets, patient_ids = preprocessor.create_sequences(
        health_records_df, patients_df, sequence_length=10
    )
    
    data_splits = preprocessor.split_data(sequences, targets, patient_ids)
    
    # Create datasets and loaders
    train_dataset = HealthDataset(**data_splits['train'])
    val_dataset = HealthDataset(**data_splits['val'])
    test_dataset = HealthDataset(**data_splits['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    model = MultiTaskTCN(input_size=13)
    trainer = ModelTrainer(model)
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=50, patience=10)
    
    # Evaluate
    test_results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.3f}")
    
    # Plot training history
    trainer.plot_training_history()