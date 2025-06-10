import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("torch_geometric not available, using SimpleGNN only")
import math


class SimpleGNN(nn.Module):
    """Simplified GNN that doesn't require torch_geometric and handles flexible input shapes"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Output dimensions
        self.output_steps = 60
        self.output_features = 5
        self.hidden_dim = config.get("HIDDEN_DIM", 256)
        
        # These will be set dynamically
        self.node_encoder = None
        self.input_dim = None
        
        # Simple attention mechanism for agent interactions
        self.agent_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get("NUM_HEADS", 8),
            dropout=config.get("DROPOUT", 0.1),
            batch_first=True
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get("DROPOUT", 0.1))
            ) for _ in range(config.get("NUM_MESSAGE_LAYERS", 3))
        ])
        
        # Ego agent decoder
        self.ego_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("DROPOUT", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_steps * self.output_features)
        )
        
    def _initialize_node_encoder(self, input_dim):
        """Initialize node encoder based on actual input dimensions"""
        if self.node_encoder is None:
            self.input_dim = input_dim
            self.node_encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.get("DROPOUT", 0.1)),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ).to(next(self.parameters()).device)
    
    def compute_distances(self, positions):
        """Compute pairwise distances between agents"""
        # positions: (batch_size, num_agents, 2)
        batch_size, num_agents, _ = positions.shape
        
        # Expand dimensions for broadcasting
        pos_i = positions.unsqueeze(2)  # (batch, num_agents, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, num_agents, 2)
        
        # Compute distances
        distances = torch.norm(pos_i - pos_j, dim=-1)  # (batch, num_agents, num_agents)
        
        return distances
    
    def create_adjacency_matrix(self, positions, distance_threshold=50.0):
        """Create adjacency matrix based on distance"""
        distances = self.compute_distances(positions)
        
        # Create adjacency matrix (1 if within threshold, 0 otherwise)
        adj_matrix = (distances <= distance_threshold).float()
        
        # Remove self-connections
        batch_size, num_agents, _ = adj_matrix.shape
        eye = torch.eye(num_agents, device=adj_matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_matrix = adj_matrix * (1 - eye)
        
        return adj_matrix
    
    def forward(self, x):
        """
        Forward pass - handles both 3D and 4D input
        Args:
            x: (batch_size, time_steps, features) or (batch_size, num_agents, time_steps, features)
        """
        # Handle different input shapes
        if len(x.shape) == 3:
            # Input is (batch_size, time_steps, features) - assume single agent (ego)
            batch_size, time_steps, features = x.shape
            
            # Use only first 5 features if available
            if features >= 5:
                x_features = x[:, :, :5]
            else:
                x_features = x
            
            # Initialize node encoder
            flattened_dim = time_steps * x_features.shape[-1]
            self._initialize_node_encoder(flattened_dim)
            
            # Flatten and encode
            node_features = x_features.reshape(batch_size, -1)
            ego_embedding = self.node_encoder(node_features)
            
        elif len(x.shape) == 4:
            # Input is (batch_size, num_agents, time_steps, features)
            batch_size, num_agents, time_steps, num_features = x.shape
            
            # Use only first 5 features if available
            if num_features >= 5:
                x_features = x[:, :, :, :5]
            else:
                x_features = x
            
            # Initialize node encoder
            flattened_dim = time_steps * x_features.shape[-1]
            self._initialize_node_encoder(flattened_dim)
            
            # Get last positions for graph construction
            last_positions = x_features[:, :, -1, :2]  # (batch_size, num_agents, 2)
            
            # Create adjacency matrix
            adj_matrix = self.create_adjacency_matrix(last_positions)
            
            # Flatten trajectory features for each agent
            node_features = x_features.reshape(batch_size, num_agents, -1)
            
            # Encode node features
            node_embeddings = self.node_encoder(node_features)  # (batch, num_agents, hidden_dim)
            
            # Apply attention to capture agent interactions
            attn_output, _ = self.agent_attention(node_embeddings, node_embeddings, node_embeddings)
            node_embeddings = node_embeddings + attn_output  # Residual connection
            
            # Message passing
            for message_layer in self.message_layers:
                # Aggregate messages from neighbors
                messages = torch.bmm(adj_matrix, node_embeddings)  # (batch, num_agents, hidden_dim)
                
                # Combine node embeddings with aggregated messages
                combined = torch.cat([node_embeddings, messages], dim=-1)
                node_embeddings = message_layer(combined) + node_embeddings  # Residual connection
            
            # Extract ego agent embedding (index 0)
            ego_embedding = node_embeddings[:, 0, :]  # (batch_size, hidden_dim)
            
        else:
            raise ValueError(f"Expected 3D or 4D input, got {len(x.shape)}D input with shape {x.shape}")
        
        # Decode trajectory
        trajectory_pred = self.ego_decoder(ego_embedding)
        trajectory_pred = trajectory_pred.view(batch_size, self.output_steps, self.output_features)
        
        return trajectory_pred

# # Updated config for GNN
# def get_gnn_config():
#     return {
#         "BATCH_SIZE": 32,  # Smaller batch size for GNN
#         "LEARNING_RATE": 0.001,
#         "EPOCHS": 100,
#         "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
#         "TEST_SIZE": 0.2,
#         "NUM_SAMPLES": 10000,
#         "DROPOUT": 0.1,
        
#         # GNN specific parameters
#         "HIDDEN_DIM": 256,
#         "NUM_GNN_LAYERS": 3,
#         "NUM_MESSAGE_LAYERS": 3,
#         "NUM_HEADS": 8,
#         "GNN_TYPE": "GCN",  # or "GAT"
#         "DISTANCE_THRESHOLD": 50.0,
#         "K_NEAREST": 10,
        
#         # Analysis parameters
#         "ANALYZE": True,
#         "ANALYZE_NUM_EXAMPLES": 100,
#     }


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional

class TransformerEncoderLayer(nn.Module):
    """Enhanced Transformer Encoder Layer with updated forward signature"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu
        
        # Additional features
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        """
        Args:
            src: Input sequence (seq_len, batch_size, d_model)
            src_mask: Optional mask for attention
            src_key_padding_mask: Optional mask for padding
            is_causal: If True, applies causal masking (ignored in our case)
        """
        # Self attention with residual
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal  # Pass through but not used in our case
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Adaptive layer norm
        shift, scale = self.adaLN(src).chunk(2, dim=-1)
        src = src * (1 + scale) + shift
        
        # Feedforward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class TrajectoryTransformer(nn.Module):
    """Transformer-based trajectory prediction model with updated encoder handling"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_features = 5  # x, y, vx, vy, heading
        self.output_features = 5
        self.output_steps = 60
        
        # Input embedding
        self.input_projection = nn.Linear(self.input_features, config["HIDDEN_DIM"])
        self.positional_encoding = PositionalEncoding(config["HIDDEN_DIM"], dropout=config["DROPOUT"])
        
        # Transformer encoder with updated layer
        encoder_layer = TransformerEncoderLayer(
            d_model=config["HIDDEN_DIM"],
            nhead=config["NUM_HEADS"],
            dim_feedforward=config.get("FF_DIM", 4 * config["HIDDEN_DIM"]),
            dropout=config["DROPOUT"]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get("NUM_TRANSFORMER_LAYERS", 4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config["HIDDEN_DIM"], config["HIDDEN_DIM"] * 2),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(config["HIDDEN_DIM"] * 2, config["HIDDEN_DIM"]),
            nn.ReLU(),
            nn.Linear(config["HIDDEN_DIM"], self.output_steps * self.output_features)
        )
        
    def forward(self, x):
        # Handle different input shapes
        if len(x.shape) == 3:  # (batch, timesteps, features)
            x = x.unsqueeze(1)  # Add agent dimension
            
        batch_size, num_agents, time_steps, features = x.shape
        
        # Process each agent independently
        all_agent_embeddings = []
        for agent_idx in range(num_agents):
            agent_features = x[:, agent_idx]  # (batch, timesteps, features)
            
            # Project and add positional encoding
            agent_emb = self.input_projection(agent_features)
            agent_emb = self.positional_encoding(agent_emb)
            
            # Transformer processing - no causal masking
            agent_emb = agent_emb.transpose(0, 1)  # (timesteps, batch, features)
            agent_emb = self.transformer_encoder(agent_emb, is_causal=False)
            agent_emb = agent_emb.transpose(0, 1)  # (batch, timesteps, features)
            
            # Aggregate over time
            agent_emb = agent_emb.mean(dim=1)  # (batch, features)
            all_agent_embeddings.append(agent_emb)
        
        # Combine agent embeddings
        if num_agents > 1:
            agent_embeddings = torch.stack(all_agent_embeddings, dim=1)  # (batch, agents, features)
            agent_embeddings = agent_embeddings.mean(dim=1)  # Simple mean pooling
        else:
            agent_embeddings = all_agent_embeddings[0]
        
        # Decode trajectory
        trajectory = self.decoder(agent_embeddings)
        trajectory = trajectory.view(batch_size, self.output_steps, self.output_features)
        
        return trajectory
    
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EnsembleModel(nn.Module):
    """Ensemble of GNN and Transformer models"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize member models
        self.gnn_model = SimpleGNN(config)
        self.transformer_model = TrajectoryTransformer(config)
        
        # Ensemble weighting
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        
        # Temperature parameter for softmax weighting
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
    def forward(self, x):
        # Get predictions from both models
        gnn_pred = self.gnn_model(x)
        transformer_pred = self.transformer_model(x)
        
        # Compute adaptive weights
        weights = F.softmax(self.ensemble_weights / self.temperature, dim=0)
        
        # Combine predictions
        combined = weights[0] * gnn_pred + weights[1] * transformer_pred
        
        return combined