import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EgoAgentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.input_size = 5         # Features per time step
        self.d_model = config.get("D_MODEL", 128)  # Transformer dimension
        self.nhead = config.get("NHEAD", 8)        # Number of attention heads
        self.num_encoder_layers = config.get("NUM_ENCODER_LAYERS", 6)
        self.num_decoder_layers = config.get("NUM_DECODER_LAYERS", 6)
        self.dim_feedforward = config.get("DIM_FEEDFORWARD", 512)
        self.dropout = config.get("DROPOUT", 0.1)
        self.input_steps = 50       # Historical steps
        self.output_steps = 60      # Future steps to predict
        self.output_size = 5        # [pos_diff_x, pos_diff_y, vel_x, vel_y, heading]
        
        # Input projection
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=200)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.output_size)
        
        # Learned query embeddings for future predictions
        self.future_queries = nn.Parameter(torch.randn(self.output_steps, self.d_model))
        
    def forward(self, x):
        # x: (batch_size, 50, 5)
        batch_size = x.size(0)
        
        # Project input features to model dimension
        src = self.input_projection(x)  # (batch_size, 50, d_model)
        src = src.transpose(0, 1)       # (50, batch_size, d_model)
        src = self.pos_encoder(src)     # Add positional encoding
        src = src.transpose(0, 1)       # (batch_size, 50, d_model)
        
        # Create target queries for future steps
        tgt = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 60, d_model)
        
        # Apply transformer
        output = self.transformer(src, tgt)  # (batch_size, 60, d_model)
        
        # Project to output features
        output = self.output_projection(output)  # (batch_size, 60, 5)
        
        return output




EgoAgentNN = EgoAgentTransformer