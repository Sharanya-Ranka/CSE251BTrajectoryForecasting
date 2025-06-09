import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GNNTrajectoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.d_input = config["D_INPUT"]
        self.d_model = config["D_MODEL"]
        self.n_head = config["N_HEAD"]
        self.dropout = config["DROPOUT"]
        
        # Input projection
        self.input_proj = nn.Linear(self.d_input, self.d_model)
        
        # Graph attention layers
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Position-wise feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 4, self.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, 2)  # 2 for x,y coordinates
        
    def forward(self, x, valid_mask=None, ego_windows=None):
        # x shape: (batch_size, seq_len, d_input)
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Graph attention
        graph_out, graph_weights = self.graph_attention(
            x, x, x,
            key_padding_mask=valid_mask if valid_mask is not None else None
        )
        x = x + F.dropout(graph_out, p=self.dropout, training=self.training)
        x = self.norm1(x)
        
        # Temporal attention
        temp_out, temp_weights = self.temporal_attention(
            x, x, x,
            key_padding_mask=valid_mask if valid_mask is not None else None
        )
        x = x + F.dropout(temp_out, p=self.dropout, training=self.training)
        x = self.norm2(x)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = x + F.dropout(ff_out, p=self.dropout, training=self.training)
        x = self.norm3(x)
        
        # Output projection
        predictions = self.output_proj(x)  # (batch_size, seq_len, 2)
        
        # Compute physical loss (position)
        physical_loss = F.mse_loss(
            predictions[..., :2],
            x[..., :2]
        )
        
        # Compute smoothness loss (velocity + acceleration)
        velocities = predictions[:, 1:, :2] - predictions[:, :-1, :2]
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        
        smoothness_loss = (
            F.mse_loss(velocities, torch.zeros_like(velocities)) +
            F.mse_loss(accelerations, torch.zeros_like(accelerations))
        )
        
        return predictions, physical_loss, smoothness_loss 