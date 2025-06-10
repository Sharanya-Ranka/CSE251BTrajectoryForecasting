# model.py - Drop-in replacement for your existing model
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EgoAgentNN(nn.Module):
    """Enhanced Transformer model with acceleration features"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Model parameters
        self.input_size = 7  # pos_x, pos_y, vel_x, vel_y, heading, acc_x, acc_y
        self.d_model = config.get("D_MODEL", 256)
        self.nhead = config.get("NHEAD", 8)
        self.num_layers = config.get("NUM_HIDDEN_LAYERS", 4)
        self.dropout = config.get("DROPOUT", 0.1)
        self.output_steps = 60
        self.output_size = 5
        
        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.output_steps * self.output_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
    
    def compute_acceleration(self, positions, velocities, dt=0.1):
        """Compute acceleration from velocity differences"""
        batch_size, seq_len, _ = velocities.shape
        
        acc = torch.zeros_like(velocities)
        if seq_len > 1:
            acc[:, 1:] = (velocities[:, 1:] - velocities[:, :-1]) / dt
            acc[:, 0] = acc[:, 1]  # Use second frame's acceleration for first frame
            
        return acc
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, 5) - [pos_diff_x, pos_diff_y, vel_x, vel_y, heading]
        batch_size, seq_len, _ = x.shape
        
        # Extract features
        pos_diff = x[:, :, :2]  # position differences
        velocities = x[:, :, 2:4]  # velocities
        heading = x[:, :, 4:5]  # heading
        
        # Compute acceleration from velocities
        acceleration = self.compute_acceleration(pos_diff, velocities)
        
        # Create enhanced input features
        enhanced_input = torch.cat([
            pos_diff,      # position differences (2D)
            velocities,    # velocities (2D)
            heading,       # heading (1D)
            acceleration   # acceleration (2D)
        ], dim=-1)  # Total: 7D
        
        # Project to model dimension
        x_proj = self.input_proj(enhanced_input)
        
        # Add positional encoding
        x_proj = x_proj.transpose(0, 1)  # (seq_len, batch, d_model)
        x_proj = self.pos_encoder(x_proj)
        x_proj = x_proj.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x_proj)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(encoded, dim=1)  # (batch, d_model)
        
        # Apply layer normalization
        pooled = self.layer_norm(pooled)
        
        # Decode to output predictions
        output = self.decoder(pooled)  # (batch, 60 * 5)
        output = output.view(batch_size, self.output_steps, self.output_size)
        
        return output

class EgoAgentEnsembleModel(nn.Module):
    """Ensemble combining LSTM and Transformer"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # LSTM branch
        self.lstm_hidden_size = 128
        self.lstm_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=7,  # Enhanced input with acceleration
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=config.get("DROPOUT", 0.1)
        )
        
        # Transformer branch (smaller version for ensemble)
        self.transformer_config = config.copy()
        self.transformer_config["D_MODEL"] = 128  # Smaller for ensemble
        self.transformer = EgoAgentNN(self.transformer_config)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(self.lstm_hidden_size + 128, 256),  # LSTM + Transformer features
            nn.GELU(),
            nn.Dropout(config.get("DROPOUT", 0.1)),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(config.get("DROPOUT", 0.1)),
            nn.Linear(128, 60 * 5)
        )
    
    def compute_acceleration(self, positions, velocities, dt=0.1):
        """Compute acceleration from velocity differences"""
        batch_size, seq_len, _ = velocities.shape
        
        acc = torch.zeros_like(velocities)
        if seq_len > 1:
            acc[:, 1:] = (velocities[:, 1:] - velocities[:, :-1]) / dt
            acc[:, 0] = acc[:, 1]
            
        return acc
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract features and compute acceleration
        pos_diff = x[:, :, :2]
        velocities = x[:, :, 2:4]
        heading = x[:, :, 4:5]
        acceleration = self.compute_acceleration(pos_diff, velocities)
        
        # Enhanced input with acceleration
        enhanced_input = torch.cat([pos_diff, velocities, heading, acceleration], dim=-1)
        
        # LSTM branch
        lstm_out, _ = self.lstm(enhanced_input)
        lstm_features = lstm_out[:, -1, :]  # Use last hidden state
        
        # Transformer branch - get features from transformer encoder
        x_proj = self.transformer.input_proj(enhanced_input)
        x_proj = x_proj.transpose(0, 1)
        x_proj = self.transformer.pos_encoder(x_proj)
        x_proj = x_proj.transpose(0, 1)
        encoded = self.transformer.transformer_encoder(x_proj)
        transformer_features = torch.mean(encoded, dim=1)  # Global average pooling
        
        # Combine features
        combined_features = torch.cat([lstm_features, transformer_features], dim=1)
        
        # Generate final predictions
        output = self.fusion(combined_features)
        output = output.view(batch_size, 60, 5)
        
        return output