import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedEgoAgentNN(nn.Module):
    """Enhanced model that works as drop-in replacement for your existing EgoAgentNN"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Enhanced input processing
        self.input_size = 5  # Start with original 5 features
        self.enhanced_input_size = 12  # Will be expanded with derived features
        self.hidden_size = 256
        self.num_layers = 3
        self.output_steps = 60
        self.output_size = 5
        
        # Feature enhancement layer
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.enhanced_input_size)
        )
        
        # Multi-agent context encoder (processes other agents)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.input_size * 49, 128),  # 49 other agents max
            nn.ReLU(),
            nn.Dropout(config.get("DROPOUT", 0.1)),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Context vector
        )
        
        # Main LSTM with enhanced features
        self.lstm = nn.LSTM(
            input_size=self.enhanced_input_size + 32,  # Enhanced features + context
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.get("DROPOUT", 0.1) if self.num_layers > 1 else 0
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=config.get("DROPOUT", 0.1),
            batch_first=True
        )
        
        # Enhanced decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.get("DROPOUT", 0.1)),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.get("DROPOUT", 0.1)),
            nn.Linear(256, self.output_steps * self.output_size)
        )
        
        # Skip connection for better gradient flow
        self.skip_connection = nn.Linear(self.hidden_size, self.output_steps * self.output_size)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(self.output_size)
        
    def enhance_features(self, ego_data):
        """Enhance the basic 5 features with derived features"""
        batch_size, seq_len, _ = ego_data.shape
        
        # Extract basic features
        pos = ego_data[:, :, :2]  # positions
        vel = ego_data[:, :, 2:4]  # velocities  
        heading = ego_data[:, :, 4:5]  # heading
        
        # Compute derived features
        # 1. Speed (magnitude of velocity)
        speed = torch.norm(vel, dim=-1, keepdim=True)
        
        # 2. Acceleration (velocity differences)
        vel_padded = F.pad(vel, (0, 0, 1, 0), mode='replicate')  # Pad first timestep
        acceleration = vel - vel_padded[:, :-1, :]
        
        # 3. Angular velocity (heading differences)
        heading_padded = F.pad(heading, (0, 0, 1, 0), mode='replicate')
        angular_vel = heading - heading_padded[:, :-1, :]
        
        # 4. Curvature approximation
        speed_safe = speed + 1e-6
        curvature = torch.abs(angular_vel) / speed_safe
        
        # Combine enhanced features
        enhanced = torch.cat([
            pos, vel, heading,  # Original 5 features
            speed,              # 1 feature
            acceleration,       # 2 features  
            angular_vel,        # 1 feature
            curvature          # 1 feature
        ], dim=-1)  # Total: 5 + 1 + 2 + 1 + 1 = 10 features
        
        # Pass through enhancement layer to get desired size
        enhanced = self.feature_enhancer(ego_data)  # (batch, seq, enhanced_input_size)
        
        return enhanced
    
    def extract_context(self, full_scene_data):
        """Extract context from other agents while handling zero-padding"""
        batch_size, num_agents, seq_len, num_features = full_scene_data.shape
        
        if num_agents > 1:
            # Get other agents (excluding ego at index 0)
            other_agents = full_scene_data[:, 1:, :, :5]  # (batch, 49, 50, 5)
            
            # Create mask for valid agents (non-zero)
            agent_mask = torch.any(other_agents != 0, dim=(2, 3))  # (batch, 49)
            
            # Flatten other agents for processing
            other_agents_flat = other_agents.reshape(batch_size, -1)  # (batch, 49*50*5)
            
            # Apply mask to zero out invalid agents
            mask_expanded = agent_mask.repeat_interleave(seq_len * 5, dim=1)  # (batch, 49*50*5)
            other_agents_masked = other_agents_flat * mask_expanded
            
            # Process through context encoder
            context = self.context_encoder(other_agents_masked)  # (batch, 32)
            
            # Expand context to match sequence length
            context = context.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, 50, 32)
        else:
            # No other agents, use zero context
            context = torch.zeros(batch_size, seq_len, 32, device=full_scene_data.device)
            
        return context
    
    def forward(self, x):
        """
        Forward pass that can handle both:
        1. Simple ego data: (batch, 50, 5) 
        2. Full scene data: (batch, 50, 50, 5) - if available
        """
        
        if len(x.shape) == 4:
            # Full scene data available
            batch_size, num_agents, seq_len, num_features = x.shape
            ego_data = x[:, 0, :, :5]  # Extract ego agent
            context = self.extract_context(x)
        elif len(x.shape) == 3:
            # Only ego data available
            batch_size, seq_len, num_features = x.shape
            ego_data = x
            context = torch.zeros(batch_size, seq_len, 32, device=x.device)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Enhance ego features
        enhanced_ego = self.enhance_features(ego_data)  # (batch, 50, enhanced_input_size)
        
        # Combine enhanced ego features with context
        combined_input = torch.cat([enhanced_ego, context], dim=-1)  # (batch, 50, enhanced_input_size + 32)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_input)  # (batch, 50, hidden_size)
        
        # Temporal attention
        attended_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)  # (batch, 50, hidden_size)
        
        # Use final timestep for prediction
        final_state = attended_out[:, -1, :]  # (batch, hidden_size)
        
        # Decoder
        decoded = self.decoder(final_state)  # (batch, 60*5)
        
        # Skip connection
        skip = self.skip_connection(final_state)  # (batch, 60*5)
        
        # Combine with residual connection
        output = decoded + skip  # Residual connection
        
        # Reshape to output format
        output = output.view(batch_size, self.output_steps, self.output_size)  # (batch, 60, 5)
        
        # Apply output normalization
        output = self.output_norm(output)
        
        return output


class ImprovedTrainingMixin:
    """Mixin class to add improved training methods to your existing trainer"""
    
    def enhanced_loss(self, pred, target, temporal_weighting=True):
        """Enhanced loss with temporal weighting and multi-component loss"""
        
        if temporal_weighting:
            # Increase weight for later predictions (more important)
            time_weights = torch.linspace(1.0, 2.0, 60).to(pred.device)
            
            # Feature weights: position > velocity > heading
            feature_weights = torch.tensor([3.0, 3.0, 1.0, 1.0, 0.5]).to(pred.device)
            
            # Combine temporal and feature weights
            weights = time_weights[:, None] * feature_weights[None, :]  # (60, 5)
            weights = weights[None, :, :]  # (1, 60, 5)
        else:
            weights = torch.ones_like(pred)
        
        # MSE Loss with weighting
        mse_loss = ((pred - target) ** 2) * weights
        
        # L1 Loss for robustness (less sensitive to outliers)
        l1_loss = torch.abs(pred - target) * weights * 0.1
        
        # Position consistency loss (adjacent timesteps should be smooth)
        pos_pred = pred[:, :, :2]  # Extract positions
        pos_target = target[:, :, :2]
        
        # Smooth trajectory loss
        pred_diff = pos_pred[:, 1:] - pos_pred[:, :-1]
        target_diff = pos_target[:, 1:] - pos_target[:, :-1]
        smooth_loss = torch.mean((pred_diff - target_diff) ** 2) * 0.1
        
        total_loss = mse_loss.mean() + l1_loss.mean() + smooth_loss
        
        return total_loss
    
    def setup_advanced_optimizer(self):
        """Setup advanced optimizer with better scheduling"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["LEARNING_RATE"],
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=self.config["LEARNING_RATE"] * 0.01
        )


# Enhanced version of your existing model that's compatible
class EgoAgentEnsembleModel(nn.Module):
    """Drop-in replacement for your existing ensemble model"""
    def __init__(self, config):
        super().__init__()
        
        # Create multiple enhanced models for ensemble
        self.models = nn.ModuleList([
            EnhancedEgoAgentNN(config) for _ in range(3)
        ])
        
        # Ensemble weighting
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        """Forward pass through ensemble"""
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Stack outputs
        stacked = torch.stack(outputs, dim=0)  # (3, batch, 60, 5)
        
        # Weighted average
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = torch.sum(stacked * weights[:, None, None, None], dim=0)
        
        return ensemble_output


# Usage instructions:
"""
To use this enhanced model as a drop-in replacement:

1. Replace your model import:
   from enhanced_model import EnhancedEgoAgentNN as EgoAgentNN
   # OR
   from enhanced_model import EgoAgentEnsembleModel

2. The model automatically handles:
   - Feature enhancement (acceleration, speed, curvature)
   - Multi-agent context (if full scene data is provided)
   - Temporal attention mechanisms
   - Better loss functions

3. Optional: Use the ImprovedTrainingMixin in your trainer:
   
   class YourTrainer(ImprovedTrainingMixin):
       def computeLoss(self, true, prediction):
           return self.enhanced_loss(prediction, true)
           
       def setUpOptimizer(self):
           self.setup_advanced_optimizer()

4. The model input can be either:
   - (batch, 50, 5) for ego-only data
   - (batch, 50, 50, 5) for full scene data with all agents
"""