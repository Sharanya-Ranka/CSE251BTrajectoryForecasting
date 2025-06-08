import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class EnhancedTrajectoryPredictor(nn.Module):
    """
    Enhanced trajectory prediction model with:
    1. Better handling of zero-padded agents
    2. Data augmentation with acceleration features
    3. Graph Neural Network for agent interactions
    4. Multi-output predictions (position, velocity, acceleration)
    5. Sliding window approach for temporal modeling
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length = 50  # Input sequence length
        self.pred_length = 60  # Prediction length
        
        # Feature dimensions
        self.raw_features = 6  # Original features
        self.augmented_features = 10  # After adding acceleration, jerk, etc.
        
        # Agent type embedding
        self.agent_type_embedding = nn.Embedding(10, config['AGENT_TYPE_DIM'])
        
        # Feature augmentation layer
        self.feature_augmenter = FeatureAugmenter(config)
        
        # Agent mask generator (to handle zero padding)
        self.mask_generator = AgentMaskGenerator()
        
        # Graph Neural Network for agent interactions
        self.gnn = AgentInteractionGNN(config)
        
        # Temporal encoder (transformer-based)
        self.temporal_encoder = TemporalEncoder(config)
        
        # Multi-head predictor
        self.predictor = MultiOutputPredictor(config)
        
        # Sliding window processor
        self.sliding_window_size = config.get('SLIDING_WINDOW_SIZE', 10)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_agents, seq_length, features)
        Returns:
            predictions: Dict with 'position', 'velocity', 'acceleration' predictions
        """
        batch_size, num_agents, seq_length, _ = x.shape
        
        # Generate agent masks (to handle zero padding)
        agent_masks = self.mask_generator(x)  # (batch_size, num_agents)
        
        # Augment features (add acceleration, jerk, etc.)
        augmented_features = self.feature_augmenter(x)  # (batch_size, num_agents, seq_length, augmented_features)
        
        # Apply sliding window processing
        windowed_features = self.apply_sliding_window(augmented_features)
        
        # Graph neural network for agent interactions
        interaction_features = self.gnn(windowed_features, agent_masks)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(interaction_features, agent_masks)
        
        # Multi-output prediction
        predictions = self.predictor(temporal_features, agent_masks)
        
        return predictions
    
    def apply_sliding_window(self, features):
        """Apply sliding window to capture local temporal patterns"""
        batch_size, num_agents, seq_length, feature_dim = features.shape
        window_size = self.sliding_window_size
        
        # Create sliding windows
        windows = []
        for i in range(seq_length - window_size + 1):
            window = features[:, :, i:i+window_size, :]
            # Aggregate window (mean, max, last)
            window_mean = torch.mean(window, dim=2)
            window_max, _ = torch.max(window, dim=2)
            window_last = window[:, :, -1, :]
            
            # Combine aggregated features
            combined = torch.cat([window_mean, window_max, window_last], dim=-1)
            windows.append(combined)
        
        # Stack windows
        windowed = torch.stack(windows, dim=2)  # (batch_size, num_agents, num_windows, feature_dim*3)
        
        return windowed


class FeatureAugmenter(nn.Module):
    """Augment raw features with derived features like acceleration, jerk"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_agents, seq_length, 6)
        Returns:
            augmented: (batch_size, num_agents, seq_length, augmented_features)
        """
        batch_size, num_agents, seq_length, _ = x.shape
        
        # Extract position and velocity
        pos = x[:, :, :, :2]  # (batch_size, num_agents, seq_length, 2)
        vel = x[:, :, :, 2:4]  # (batch_size, num_agents, seq_length, 2)
        heading = x[:, :, :, 4:5]  # (batch_size, num_agents, seq_length, 1)
        obj_type = x[:, :, :, 5:6]  # (batch_size, num_agents, seq_length, 1)
        
        # Calculate acceleration (derivative of velocity)
        acc = torch.zeros_like(vel)
        acc[:, :, 1:, :] = vel[:, :, 1:, :] - vel[:, :, :-1, :]
        
        # Calculate jerk (derivative of acceleration)
        jerk = torch.zeros_like(vel)
        jerk[:, :, 1:, :] = acc[:, :, 1:, :] - acc[:, :, :-1, :]
        
        # Calculate speed (magnitude of velocity)
        speed = torch.norm(vel, dim=-1, keepdim=True)
        
        # Calculate angular velocity (change in heading)
        angular_vel = torch.zeros_like(heading)
        angular_vel[:, :, 1:, :] = heading[:, :, 1:, :] - heading[:, :, :-1, :]
        
        # Combine all features
        augmented = torch.cat([pos, vel, acc, jerk, heading, angular_vel, speed, obj_type], dim=-1)
        
        return augmented


class AgentMaskGenerator(nn.Module):
    """Generate masks to identify valid (non-padded) agents"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_agents, seq_length, features)
        Returns:
            masks: (batch_size, num_agents) - 1 for valid agents, 0 for padded
        """
        # An agent is considered valid if it has non-zero position variance
        pos = x[:, :, :, :2]  # Extract positions
        pos_var = torch.var(pos, dim=2)  # Variance across time
        pos_var_sum = torch.sum(pos_var, dim=-1)  # Sum of x and y variances
        
        # Agent is valid if it has some movement (variance > small threshold)
        masks = (pos_var_sum > 1e-6).float()
        
        return masks


class AgentInteractionGNN(nn.Module):
    """Graph Neural Network to model agent interactions"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.get('GNN_FEATURE_DIM', 64)
        self.hidden_dim = config.get('GNN_HIDDEN_DIM', 128)
        self.num_layers = config.get('GNN_NUM_LAYERS', 3)
        
        # Feature projection
        self.feature_proj = nn.Linear(30, self.feature_dim)  # 10 features * 3 (from sliding window)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.feature_dim, self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.feature_dim)
        
    def forward(self, x, masks):
        """
        Args:
            x: (batch_size, num_agents, num_windows, feature_dim*3)
            masks: (batch_size, num_agents)
        Returns:
            interaction_features: (batch_size, num_agents, num_windows, feature_dim)
        """
        batch_size, num_agents, num_windows, _ = x.shape
        
        # Project features
        x = self.feature_proj(x)  # (batch_size, num_agents, num_windows, feature_dim)
        
        # Process each time window
        outputs = []
        for t in range(num_windows):
            xt = x[:, :, t, :]  # (batch_size, num_agents, feature_dim)
            
            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                xt = gnn_layer(xt, masks)
            
            xt = self.output_proj(xt)
            outputs.append(xt)
        
        # Stack outputs
        interaction_features = torch.stack(outputs, dim=2)
        
        return interaction_features


class GNNLayer(nn.Module):
    """Single GNN layer with attention-based message passing"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Message functions
        self.message_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, masks):
        """
        Args:
            x: (batch_size, num_agents, feature_dim)
            masks: (batch_size, num_agents)
        Returns:
            updated_x: (batch_size, num_agents, feature_dim)
        """
        batch_size, num_agents, feature_dim = x.shape
        
        # Compute pairwise interactions
        x_expanded = x.unsqueeze(2).expand(-1, -1, num_agents, -1)  # (batch_size, num_agents, num_agents, feature_dim)
        x_transposed = x.unsqueeze(1).expand(-1, num_agents, -1, -1)  # (batch_size, num_agents, num_agents, feature_dim)
        
        # Concatenate for message computation
        pairs = torch.cat([x_expanded, x_transposed], dim=-1)  # (batch_size, num_agents, num_agents, feature_dim*2)
        
        # Compute messages
        messages = self.message_net(pairs)  # (batch_size, num_agents, num_agents, hidden_dim)
        
        # Compute attention weights
        attention_logits = self.attention_net(pairs).squeeze(-1)  # (batch_size, num_agents, num_agents)
        
        # Apply masks to attention (mask out padded agents)
        mask_expanded = masks.unsqueeze(1).expand(-1, num_agents, -1)  # (batch_size, num_agents, num_agents)
        attention_logits = attention_logits.masked_fill(~mask_expanded.bool(), -float('inf'))
        
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = attention_weights.masked_fill(~mask_expanded.bool(), 0)
        
        # Aggregate messages
        aggregated_messages = torch.sum(messages * attention_weights.unsqueeze(-1), dim=2)  # (batch_size, num_agents, hidden_dim)
        
        # Update node features
        combined = torch.cat([x, aggregated_messages], dim=-1)
        updated_x = self.update_net(combined)
        
        # Apply residual connection
        updated_x = x + updated_x
        
        return updated_x


class TemporalEncoder(nn.Module):
    """Temporal encoder using transformer architecture"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.get('TEMPORAL_D_MODEL', 256)
        self.n_heads = config.get('TEMPORAL_N_HEADS', 8)
        self.num_layers = config.get('TEMPORAL_NUM_LAYERS', 4)
        
        # Input projection
        self.input_proj = nn.Linear(config.get('GNN_FEATURE_DIM', 64), self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            batch_first=True,
            dropout=config.get('DROPOUT', 0.1)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
    def forward(self, x, masks):
        """
        Args:
            x: (batch_size, num_agents, num_windows, feature_dim)
            masks: (batch_size, num_agents)
        Returns:
            encoded: (batch_size, num_agents, num_windows, d_model)
        """
        batch_size, num_agents, num_windows, feature_dim = x.shape
        
        # Focus on ego agent (agent 0) and its context
        ego_features = x[:, 0, :, :]  # (batch_size, num_windows, feature_dim)
        
        # Project to model dimension
        ego_projected = self.input_proj(ego_features)  # (batch_size, num_windows, d_model)
        
        # Add positional encoding
        ego_encoded = self.pos_encoding(ego_projected)
        
        # Apply transformer
        ego_output = self.transformer(ego_encoded)  # (batch_size, num_windows, d_model)
        
        # Prepare output (focus on ego agent)
        output = torch.zeros(batch_size, num_agents, num_windows, self.d_model, device=x.device)
        output[:, 0, :, :] = ego_output
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class MultiOutputPredictor(nn.Module):
    """Multi-output predictor for position, velocity, and acceleration"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.get('TEMPORAL_D_MODEL', 256)
        self.pred_length = 60
        
        # Separate predictors for different outputs
        self.position_predictor = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(config.get('DROPOUT', 0.1)),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.pred_length * 2)  # x, y for each time step
        )
        
        self.velocity_predictor = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(config.get('DROPOUT', 0.1)),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.pred_length * 2)  # vx, vy for each time step
        )
        
        self.acceleration_predictor = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(config.get('DROPOUT', 0.1)),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.pred_length * 2)  # ax, ay for each time step
        )
        
    def forward(self, x, masks):
        """
        Args:
            x: (batch_size, num_agents, num_windows, d_model)
            masks: (batch_size, num_agents)
        Returns:
            predictions: Dict with 'position', 'velocity', 'acceleration'
        """
        # Focus on ego agent
        ego_features = x[:, 0, :, :]  # (batch_size, num_windows, d_model)
        
        # Global pooling across time windows
        ego_global = torch.mean(ego_features, dim=1)  # (batch_size, d_model)
        
        # Predict different outputs
        pos_pred = self.position_predictor(ego_global)  # (batch_size, pred_length * 2)
        vel_pred = self.velocity_predictor(ego_global)  # (batch_size, pred_length * 2)
        acc_pred = self.acceleration_predictor(ego_global)  # (batch_size, pred_length * 2)
        
        # Reshape predictions
        batch_size = pos_pred.shape[0]
        pos_pred = pos_pred.view(batch_size, self.pred_length, 2)
        vel_pred = vel_pred.view(batch_size, self.pred_length, 2)
        acc_pred = acc_pred.view(batch_size, self.pred_length, 2)
        
        return {
            'position': pos_pred,
            'velocity': vel_pred,
            'acceleration': acc_pred
        }


# Enhanced trainer class
class EnhancedTrainer:
    """Enhanced trainer with multi-output loss and better evaluation"""
    
    def __init__(self, config):
        self.config = config
        
    def compute_multi_loss(self, predictions, targets):
        """
        Compute weighted loss for multiple outputs
        """
        pos_target = targets  # Assuming targets are positions
        pos_pred = predictions['position']
        
        # Position loss (primary)
        pos_loss = F.mse_loss(pos_pred, pos_target)
        
        # Velocity loss (if available)
        vel_loss = 0
        if 'velocity' in predictions:
            # Compute velocity from position differences
            vel_target = torch.zeros_like(pos_target)
            vel_target[:, 1:, :] = pos_target[:, 1:, :] - pos_target[:, :-1, :]
            vel_loss = F.mse_loss(predictions['velocity'], vel_target)
        
        # Acceleration loss (if available)
        acc_loss = 0
        if 'acceleration' in predictions:
            # Compute acceleration from velocity differences
            if 'velocity' in predictions:
                vel_pred = predictions['velocity']
                acc_target = torch.zeros_like(vel_pred)
                acc_target[:, 1:, :] = vel_pred[:, 1:, :] - vel_pred[:, :-1, :]
                acc_loss = F.mse_loss(predictions['acceleration'], acc_target)
        
        # Weighted combination
        total_loss = (
            self.config.get('POS_WEIGHT', 1.0) * pos_loss +
            self.config.get('VEL_WEIGHT', 0.1) * vel_loss +
            self.config.get('ACC_WEIGHT', 0.01) * acc_loss
        )
        
        return total_loss, {'pos_loss': pos_loss, 'vel_loss': vel_loss, 'acc_loss': acc_loss}


# Example configuration
def get_enhanced_config():
    return {
        # Basic training parameters
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 100,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.1,
        "NUM_SAMPLES": 10000,
        "DROPOUT": 0.1,
        
        # Model architecture parameters
        "AGENT_TYPE_DIM": 16,
        "SLIDING_WINDOW_SIZE": 10,
        
        # GNN parameters
        "GNN_FEATURE_DIM": 64,
        "GNN_HIDDEN_DIM": 128,
        "GNN_NUM_LAYERS": 3,
        
        # Temporal encoder parameters
        "TEMPORAL_D_MODEL": 256,
        "TEMPORAL_N_HEADS": 8,
        "TEMPORAL_NUM_LAYERS": 4,
        
        # Loss weights
        "POS_WEIGHT": 1.0,
        "VEL_WEIGHT": 0.1,
        "ACC_WEIGHT": 0.01,
        
        # Analysis parameters
        "ANALYZE": True,
        "ANALYZE_NUM_BATCHES": 50,
    }


def main():
    # Load configuration
    config = get_enhanced_config()
    
    # Create model
    model = EnhancedTrajectoryPredictor(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example input data (batch_size=2, num_agents=5, seq_length=50, features=6)
    batch_size = 2
    num_agents = 5
    seq_length = 50
    features = 6
    
    # Create random input data (positions, velocities, heading, object type)
    x = torch.randn(batch_size, num_agents, seq_length, features)
    
    # Set some agents to zero (padding)
    x[0, 3:, :, :] = 0  # First batch, last 2 agents are padded
    x[1, 2:, :, :] = 0  # Second batch, last 3 agents are padded
    
    # Forward pass
    predictions = model(x)
    
    # Print predictions
    print("\nPredictions:")
    for key, value in predictions.items():
        print(f"{key}: {value.shape}")
    
    # Create trainer and compute loss (with dummy targets)
    trainer = EnhancedTrainer(config)
    targets = torch.randn(batch_size, 60, 2)  # Random targets for 60 timesteps
    total_loss, loss_components = trainer.compute_multi_loss(predictions, targets)
    
    print("\nLosses:")
    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in loss_components.items():
        print(f"{key}: {value.item():.4f}")


if __name__ == "__main__":
    main()