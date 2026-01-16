import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for time-series data"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TimeGapEncoding(nn.Module):
    """Encoding for irregular time gaps between visits"""
    
    def __init__(self, d_model: int):
        super(TimeGapEncoding, self).__init__()
        self.d_model = d_model
        self.time_embedding = nn.Linear(1, d_model)
    
    def forward(self, time_gaps):
        # time_gaps shape: (batch_size, seq_len, 1)
        # Normalize time gaps (log transform for better distribution)
        normalized_gaps = torch.log(time_gaps + 1)  # +1 to avoid log(0)
        return self.time_embedding(normalized_gaps)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output, attention_weights

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class TimeSeriesTransformer(nn.Module):
    """Transformer for time-series health data"""
    
    def __init__(self, input_size: int, d_model: int = 128, n_heads: int = 8, 
                 n_layers: int = 4, d_ff: int = 512, dropout: float = 0.1, 
                 max_seq_len: int = 100):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size - 1, d_model)  # -1 for time gap
        
        # Encodings
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.time_gap_encoding = TimeGapEncoding(d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def create_padding_mask(self, x):
        """Create mask for padded sequences"""
        # Assume padding is all zeros
        mask = (x.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Extract time gaps (last feature) and health features
        health_features = x[:, :, :-1]  # All except last column
        time_gaps = x[:, :, -1:].unsqueeze(-1)  # Last column (time gaps)
        
        # Project health features to model dimension
        x = self.input_projection(health_features)  # (batch_size, seq_len, d_model)
        
        # Add time gap encoding
        time_encoding = self.time_gap_encoding(time_gaps)
        x = x + time_encoding
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Create padding mask
        mask = self.create_padding_mask(health_features)
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights

class MultiTaskTransformer(nn.Module):
    """Multi-task Transformer for disease prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super(MultiTaskTransformer, self).__init__()
        
        # Transformer backbone
        self.transformer = TimeSeriesTransformer(
            input_size=input_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Global attention pooling (learnable aggregation)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, 128),
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
        # Apply transformer
        transformer_out, attention_weights = self.transformer(x)
        # transformer_out shape: (batch_size, seq_len, d_model)
        
        # Attention-based pooling
        attention_scores = self.attention_pool(transformer_out)  # (batch_size, seq_len, 1)
        pooled_features = torch.sum(transformer_out * attention_scores, dim=1)  # (batch_size, d_model)
        
        # Shared features
        shared_features = self.shared_fc(pooled_features)
        
        # Disease predictions
        diabetes_pred = self.diabetes_head(shared_features)
        heart_pred = self.heart_disease_head(shared_features)
        kidney_pred = self.kidney_disease_head(shared_features)
        
        return {
            'diabetes': diabetes_pred.squeeze(-1),
            'heart_disease': heart_pred.squeeze(-1),
            'kidney_disease': kidney_pred.squeeze(-1),
            'shared_features': shared_features,
            'attention_weights': attention_weights,
            'pooling_attention': attention_scores.squeeze(-1)
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
    model = MultiTaskTransformer(
        input_size=input_size,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1
    )
    
    # Forward pass
    outputs = model(x)
    
    print("Model output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} attention layers")
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")