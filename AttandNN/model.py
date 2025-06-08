import torch
from torch import nn
import torch.nn.functional as F
import math


class AttentionAndNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length = 110
        self.num_modes = config.get('NUM_MODES', 6)  # Multi-modal predictions

        # Enhanced embeddings
        self.queriesEmbedding = nn.Embedding(config['NUM_QUERIES'], config['D_MODEL'])
        self.agent_type_embedding = nn.Embedding(4, config['D_MODEL'])  # ego, vehicle, pedestrian, cyclist
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(self.seq_length, config['D_MODEL'])
        
        # Input transformation with better normalization
        self.upscaleTransform = nn.Sequential(
            nn.Linear(config['D_INPUT'], config['D_MODEL']),
            nn.LayerNorm(config['D_MODEL']),
            nn.GELU(),
            nn.Dropout(config['DROPOUT'])
        )
        
        # Map/lane encoder for contextual information
        self.lane_encoder = nn.Sequential(
            nn.Linear(2, 64),  # lane centerline coords
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, config['D_MODEL']),
            nn.GELU(),
            nn.Dropout(config['DROPOUT'])
        )
        
        # Enhanced transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["D_MODEL"],
            nhead=config["N_HEAD"],
            dim_feedforward=config["D_MODEL"] * 4,  # Larger FFN
            batch_first=True,
            dropout=config["DROPOUT"],
            activation='gelu',
            norm_first=True  # Pre-norm architecture
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config["NUM_LAYERS"],
            norm=nn.LayerNorm(config["D_MODEL"])
        )
        self.encoder = transformer_encoder
        
        # Social attention for agent interactions
        self.social_attention = nn.MultiheadAttention(
            config['D_MODEL'], 
            config['N_HEAD'], 
            batch_first=True,
            dropout=config['DROPOUT']
        )
        
        # Multi-modal prediction head
        ffnn_config = {
            "D_INPUT": config['NUM_QUERIES'] * config['D_MODEL'] + config['D_EGO_FFNN_INPUT'] * 50,
            "D_HIDDEN": config["FFNN_D_HIDDEN"],
            "D_OUTPUT": 60 * 2 * self.num_modes,  # Multi-modal outputs
            "DROPOUT": config["DROPOUT"],
            "NUM_HIDDEN_LAYERS": config.get("FFNN_NUM_HIDDEN_LAYERS", 2)  # Deeper network
        }
        self.prediction_nn = FeedForwardNN(ffnn_config)
        
        # Mode probability head
        self.mode_prob_head = nn.Sequential(
            nn.Linear(config['NUM_QUERIES'] * config['D_MODEL'] + config['D_EGO_FFNN_INPUT'] * 50, 
                     config["FFNN_D_HIDDEN"]),
            nn.GELU(),
            nn.Dropout(config['DROPOUT']),
            nn.Linear(config["FFNN_D_HIDDEN"], self.num_modes)
        )
        
    def _create_positional_encoding(self, seq_len, d_model):
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, inp, lane_data=None, agent_types=None):
        device = self.config['DEVICE']
        num_queries = self.config['NUM_QUERIES']
        dmodel = self.config['D_MODEL']
        batch_size = inp.size()[0]
        
        # Enhanced queries with learnable embeddings
        queries = torch.reshape(
            self.queriesEmbedding(torch.arange(num_queries).to(device)), 
            (1, num_queries, dmodel)
        ).expand((batch_size, -1, -1))
        
        # Process input with positional encoding
        other_agent_inp = self.upscaleTransform(torch.flatten(inp.transpose(1, 2), start_dim=2))
        
        # Add positional encoding
        seq_len = other_agent_inp.size(1)
        other_agent_inp = other_agent_inp + self.positional_encoding[:, :seq_len, :].to(device)
        
        # Add agent type embeddings if provided
        if agent_types is not None:
            agent_type_emb = self.agent_type_embedding(agent_types)
            other_agent_inp = other_agent_inp + agent_type_emb
        
        # Include lane information if available
        if lane_data is not None:
            lane_features = self.lane_encoder(lane_data)  # Shape: (batch, num_lanes, d_model)
            # Concatenate lane features to the input
            precompact_inp = torch.cat((queries, other_agent_inp, lane_features), axis=1)
        else:
            precompact_inp = torch.cat((queries, other_agent_inp), axis=1)

        # Enhanced encoding with social attention
        encoded_features = self.encoder(precompact_inp)
        
        # Extract query features and apply social attention
        query_features = encoded_features[:, :num_queries]
        context_features = encoded_features[:, num_queries:]
        
        # Social attention between agents
        if context_features.size(1) > 0:
            social_attended, _ = self.social_attention(
                query_features, context_features, context_features
            )
            query_features = query_features + social_attended
        
        compact_inp = torch.flatten(query_features, start_dim=1)

        # Only need xpos, ypos, xvel and yvel from ego agent
        ego_inp = torch.flatten(inp[:, 0, :, :4], start_dim=1)

        full_pred_inp = torch.cat((ego_inp, compact_inp), axis=1)

        # Multi-modal predictions
        op = self.prediction_nn(full_pred_inp)
        mode_probs = F.softmax(self.mode_prob_head(full_pred_inp), dim=-1)

        # Reshape to multi-modal trajectories
        final_op = torch.reshape(op, (-1, self.num_modes, 60, 2))

        return final_op, mode_probs


class FeedForwardNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input projection layer with improved normalization
        self.input_proj = nn.Sequential(
            nn.Linear(config["D_INPUT"], config["D_HIDDEN"]),
            nn.LayerNorm(config["D_HIDDEN"]),
            nn.GELU(),
            nn.Dropout(config["DROPOUT"])
        )

        # Enhanced residual blocks
        self.hidden_layers = nn.ModuleList()
        num_hidden_layers = config.get("NUM_HIDDEN_LAYERS", 1)
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                ResidualBlock(
                    config["D_HIDDEN"],
                    config["DROPOUT"]
                )
            )

        # Output layer with better initialization
        self.output_layer = nn.Sequential(
            nn.LayerNorm(config["D_HIDDEN"]),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(config["D_HIDDEN"], config["D_OUTPUT"])
        )
        
        # Initialize output layer with smaller weights for stable training
        nn.init.xavier_uniform_(self.output_layer[-1].weight, gain=0.1)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.hidden_layers:
            x = block(x)

        # Output layer
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),  # Pre-norm
            nn.Linear(dim, dim * 2),  # Expand
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),  # Contract
            nn.Dropout(dropout_rate)
        )
        
        # Scale residual connection
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        # Pre-norm residual connection with learned scaling
        return x + self.scale * self.block(x)


# Additional utility functions for training
class MultiModalLoss(nn.Module):
    """Winner-takes-all loss for multi-modal predictions"""
    def __init__(self, num_modes=6, reduction='mean'):
        super().__init__()
        self.num_modes = num_modes
        self.reduction = reduction
    
    def forward(self, pred_trajectories, mode_probs, target_trajectory):
        """
        pred_trajectories: (batch, num_modes, 60, 2)
        mode_probs: (batch, num_modes)
        target_trajectory: (batch, 60, 2)
        """
        batch_size = pred_trajectories.size(0)
        
        # Compute L2 distance for each mode
        target_expanded = target_trajectory.unsqueeze(1).expand(-1, self.num_modes, -1, -1)
        distances = torch.norm(pred_trajectories - target_expanded, dim=-1)  # (batch, modes, 60)
        trajectory_losses = distances.mean(dim=-1)  # Average over time steps
        
        # Winner-takes-all: select best mode for each sample
        best_modes = torch.argmin(trajectory_losses, dim=1)
        best_trajectory_loss = trajectory_losses[torch.arange(batch_size), best_modes]
        
        # Mode probability loss (cross-entropy with best mode as target)
        mode_loss = F.cross_entropy(mode_probs, best_modes)
        
        total_loss = best_trajectory_loss.mean() + 0.1 * mode_loss
        
        return total_loss