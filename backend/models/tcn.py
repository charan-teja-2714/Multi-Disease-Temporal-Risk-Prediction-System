import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CausalConv1d(nn.Module):
    """Causal 1D convolution to ensure no future information leakage"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, **kwargs):
        super(CausalConv1d, self).__init__()
        
        # Calculate padding to maintain sequence length
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=self.padding, dilation=dilation, **kwargs
        )
    
    def forward(self, x):
        # Apply convolution and remove future information
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out

class TemporalBlock(nn.Module):
    """Basic building block of TCN with residual connections"""
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        # Two causal convolutions with different dilations
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
        
        # Normalization and activation
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        self.conv1.conv.weight.data.normal_(0, 0.01)
        self.conv2.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        # First convolution block
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling"""
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 2, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, 
                dilation=dilation_size, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MultiTaskTCN(nn.Module):
    """Multi-task TCN for predicting multiple diseases"""
    
    def __init__(self, input_size: int, tcn_channels: List[int] = [64, 64, 64], 
                 kernel_size: int = 3, dropout: float = 0.2, num_diseases: int = 3):
        super(MultiTaskTCN, self).__init__()
        
        # Temporal Convolutional Network backbone
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Global average pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Disease-specific heads
        self.diabetes_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.heart_disease_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.kidney_disease_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # TCN expects: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply TCN
        tcn_out = self.tcn(x)  # (batch_size, tcn_channels[-1], sequence_length)
        
        # Global pooling to get fixed representation
        pooled = self.global_pool(tcn_out).squeeze(-1)  # (batch_size, tcn_channels[-1])
        
        # Shared features
        shared_features = self.shared_fc(pooled)
        
        # Disease predictions
        diabetes_pred = self.diabetes_head(shared_features)
        heart_pred = self.heart_disease_head(shared_features)
        kidney_pred = self.kidney_disease_head(shared_features)
        
        return {
            'diabetes': diabetes_pred.squeeze(-1),
            'heart_disease': heart_pred.squeeze(-1),
            'kidney_disease': kidney_pred.squeeze(-1),
            'shared_features': shared_features  # For explainability
        }

# Usage example
if __name__ == "__main__":
    # Test the model
    batch_size = 32
    sequence_length = 10
    input_size = 13  # 11 health features + 2 time features
    
    # Create sample data
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # Initialize model
    model = MultiTaskTCN(
        input_size=input_size,
        tcn_channels=[64, 64, 64],
        kernel_size=3,
        dropout=0.2
    )
    
    # Forward pass
    outputs = model(x)
    
    print("Model output shapes:")
    for disease, pred in outputs.items():
        if disease != 'shared_features':
            print(f"{disease}: {pred.shape}")
    
    print(f"Shared features: {outputs['shared_features'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")