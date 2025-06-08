import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random


# ============================================================================
# DATA AUGMENTATION AND DATASET CLASSES
# ============================================================================

class AugmentedTrajectoryDataset(Dataset):
    """
    Dataset wrapper that applies data augmentation techniques for trajectory prediction
    """
    
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config
        self.augment_prob = config.get('AUGMENT_PROB', 0.5)
        
        # Augmentation parameters
        self.noise_std = config.get('NOISE_STD', 0.01)
        self.rotation_range = config.get('ROTATION_RANGE', 0.1)  # radians
        self.scale_range = config.get('SCALE_RANGE', (0.95, 1.05))
        self.time_shift_range = config.get('TIME_SHIFT_RANGE', 3)  # time steps
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base data
        X, Y, indices = self.base_dataset[idx]
        
        # Apply augmentation with probability
        if random.random() < self.augment_prob and hasattr(self.base_dataset, 'training') and self.base_dataset.training:
            X, Y = self.apply_augmentation(X, Y)
        
        return X, Y, indices
    
    def apply_augmentation(self, X, Y):
        """Apply various augmentation techniques"""
        
        # Choose augmentation techniques randomly
        augmentations = []
        
        if random.random() < 0.3:
            augmentations.append('noise')
        if random.random() < 0.2:
            augmentations.append('rotation')
        if random.random() < 0.2:
            augmentations.append('scale')
        if random.random() < 0.1:
            augmentations.append('time_shift')
        
        # Apply augmentations
        for aug in augmentations:
            if aug == 'noise':
                X, Y = self.add_noise(X, Y)
            elif aug == 'rotation':
                X, Y = self.apply_rotation(X, Y)
            elif aug == 'scale':
                X, Y = self.apply_scaling(X, Y)
            elif aug == 'time_shift':
                X = self.apply_time_shift(X)
        
        return X, Y
    
    def add_noise(self, X, Y):
        """Add Gaussian noise to positions and velocities"""
        noise_X = torch.randn_like(X[:, :, :4]) * self.noise_std
        noise_Y = torch.randn_like(Y) * self.noise_std
        
        X_aug = X.clone()
        X_aug[:, :, :4] += noise_X  # Add noise to position and velocity
        
        Y_aug = Y.clone()
        Y_aug += noise_Y
        
        return X_aug, Y_aug
    
    def apply_rotation(self, X, Y):
        """Apply random rotation to the scene"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix
        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=X.dtype)
        
        X_aug = X.clone()
        Y_aug = Y.clone()
        
        # Rotate positions
        pos = X_aug[:, :, :2]  # (num_agents, seq_len, 2)
        pos_flat = pos.reshape(-1, 2)  # (num_agents * seq_len, 2)
        pos_rot = torch.matmul(pos_flat, R.T)
        X_aug[:, :, :2] = pos_rot.reshape(pos.shape)
        
        # Rotate velocities
        vel = X_aug[:, :, 2:4]
        vel_flat = vel.reshape(-1, 2)
        vel_rot = torch.matmul(vel_flat, R.T)
        X_aug[:, :, 2:4] = vel_rot.reshape(vel.shape)
        
        # Rotate target positions
        Y_flat = Y_aug.reshape(-1, 2)
        Y_rot = torch.matmul(Y_flat, R.T)
        Y_aug = Y_rot.reshape(Y_aug.shape)
        
        # Update heading
        X_aug[:, :, 4] += angle
        
        return X_aug, Y_aug
    
    def apply_scaling(self, X, Y):
        """Apply random scaling to the scene"""
        scale = random.uniform(*self.scale_range)
        
        X_aug = X.clone()
        Y_aug = Y.clone()
        
        # Scale positions and velocities
        X_aug[:, :, :4] *= scale
        Y_aug *= scale
        
        return X_aug, Y_aug
    
    def apply_time_shift(self, X):
        """Apply random time shift to create temporal variation"""
        if X.shape[1] <= self.time_shift_range * 2:
            return X  # Skip if sequence too short
        
        shift = random.randint(-self.time_shift_range, self.time_shift_range)
        
        if shift > 0:
            # Shift forward: pad beginning with first frame
            first_frame = X[:, :1, :].expand(-1, shift, -1)
            X_shifted = torch.cat([first_frame, X[:, :-shift, :]], dim=1)
        elif shift < 0:
            # Shift backward: pad end with last frame
            last_frame = X[:, -1:, :].expand(-1, -shift, -1)
            X_shifted = torch.cat([X[:, -shift:, :], last_frame], dim=1)
        else:
            X_shifted = X
        
        return X_shifted


class MultiModalTargetDataset(Dataset):
    """
    Dataset that provides multiple prediction targets (position, velocity, acceleration)
    """
    
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        X, Y, indices = self.base_dataset[idx]
        
        # Create multi-modal targets
        targets = self._create_multimodal_targets(Y)
        
        return X, targets, indices
    
    def _create_multimodal_targets(self, Y):
        """Create position, velocity, and acceleration targets"""
        # Y is position trajectory (seq_len, 2)
        
        # Position target (already provided)
        pos_target = Y
        
        # Velocity target (derivative of position)
        vel_target = torch.zeros_like(Y)
        vel_target[1:, :] = Y[1:, :] - Y[:-1, :]
        
        # Acceleration target (derivative of velocity)
        acc_target = torch.zeros_like(Y)
        acc_target[1:, :] = vel_target[1:, :] - vel_target[:-1, :]
        
        return {
            'position': pos_target,
            'velocity': vel_target,
            'acceleration': acc_target
        }


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class FeatureAugmenter(nn.Module):
    def forward(self, x):
        batch_size, num_agents, seq_length, _ = x.shape
        
        # Extract features - adjust indices based on your actual input format
        pos = x[:, :, :, :2]    # x,y position
        vel = x[:, :, :, 2:4]   # vx,vy velocity
        heading = x[:, :, :, 4:5]  # heading angle
        obj_type = x[:, :, :, 5:6]  # object type
        # Assuming remaining features are extras you want to keep
        extras = x[:, :, :, 6:]  # any additional features
        
        # Calculate derived features
        acc = torch.zeros_like(vel)
        acc[:, :, 1:, :] = vel[:, :, 1:, :] - vel[:, :, :-1, :]
        
        jerk = torch.zeros_like(vel)
        jerk[:, :, 1:, :] = acc[:, :, 1:, :] - acc[:, :, :-1, :]
        
        speed = torch.norm(vel, dim=-1, keepdim=True)
        angular_vel = torch.zeros_like(heading)
        angular_vel[:, :, 1:, :] = heading[:, :, 1:, :] - heading[:, :, :-1, :]
        
        # Combine all features (including original extras)
        augmented = torch.cat([
            pos, vel, acc, jerk, 
            heading, angular_vel, 
            speed, obj_type, extras
        ], dim=-1)
        
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


class AgentInteractionGNN(nn.Module):
    """Graph Neural Network to model agent interactions"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.get('GNN_FEATURE_DIM', 64)
        self.hidden_dim = config.get('GNN_HIDDEN_DIM', 128)
        self.num_layers = config.get('GNN_NUM_LAYERS', 3)
        
        # Feature projection
        #self.feature_proj = nn.Linear(36, self.feature_dim)  # 10 features * 3 (from sliding window)
        self.feature_proj = nn.Linear(45, config.get('GNN_FEATURE_DIM', 64))
        
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


# ============================================================================
# MAIN MODEL
# ============================================================================

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
        self.raw_features = 9  # Original features
        self.augmented_features = 15  # After adding acceleration, jerk, etc.
        
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


# ============================================================================
# TRAINER CLASSES
# ============================================================================

class EnhancedTrainer:
    """Enhanced trainer with multi-output loss and better evaluation"""
    
    def __init__(self, config):
        self.config = config
        
    def compute_multi_loss(self, predictions, targets):
        """
        Compute weighted loss for multiple outputs
        """
        # Handle both regular targets (position only) and multimodal targets
        if isinstance(targets, dict):
            pos_target = targets['position']
            vel_target = targets.get('velocity')
            acc_target = targets.get('acceleration')
        else:
            pos_target = targets
            vel_target = None
            acc_target = None
        
        pos_pred = predictions['position']
        
        # Position loss (primary)
        pos_loss = F.mse_loss(pos_pred, pos_target)
        
        # Velocity loss (if available)
        vel_loss = 0
        if 'velocity' in predictions and vel_target is not None:
            vel_loss = F.mse_loss(predictions['velocity'], vel_target)
        elif 'velocity' in predictions:
            # Compute velocity from position differences
            vel_target_computed = torch.zeros_like(pos_target)
            vel_target_computed[:, 1:, :] = pos_target[:, 1:, :] - pos_target[:, :-1, :]
            vel_loss = F.mse_loss(predictions['velocity'], vel_target_computed)
        
        # Acceleration loss (if available)
        acc_loss = 0
        if 'acceleration' in predictions and acc_target is not None:
            acc_loss = F.mse_loss(predictions['acceleration'], acc_target)
        elif 'acceleration' in predictions:
            # Compute acceleration from velocity differences
            if 'velocity' in predictions:
                vel_pred = predictions['velocity']
                acc_target_computed = torch.zeros_like(vel_pred)
                acc_target_computed[:, 1:, :] = vel_pred[:, 1:, :] - vel_pred[:, :-1, :]
                acc_loss = F.mse_loss(predictions['acceleration'], acc_target_computed)
        
        # Weighted combination
        total_loss = (
            self.config.get('POS_WEIGHT', 1.0) * pos_loss +
            self.config.get('VEL_WEIGHT', 0.1) * vel_loss +
            self.config.get('ACC_WEIGHT', 0.01) * acc_loss
        )
        
        return total_loss, {'pos_loss': pos_loss, 'vel_loss': vel_loss, 'acc_loss': acc_loss}


class EnhancedAttentionAndNNTrainer:
    def __init__(self, config):
        self.config = config
        self.enhanced_trainer = EnhancedTrainer(config)

    def performPipeline(self):
        self.setUpData()
        self.setUpModel()
        self.setUpOptimizer()
        return self.train()

    def setUpData(self):
        total_indices = np.arange(self.config["NUM_SAMPLES"])
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )

        train_config = {
            "INDICES": train_indices,
            "DATA_FILENAME": "train.npz",
        }
        test_config = {"INDICES": test_indices, "DATA_FILENAME": "train.npz"}

        # You'll need to import your existing dataset class here
        # For now, we'll assume it exists
        try:
            import AttentionAndNN.data as dataset
            self.train_dataset = dataset.AllAgentsNormalizedDataset(train_config)
            self.test_dataset = dataset.AllAgentsNormalizedDataset(test_config)
        except ImportError:
            print("Warning: Could not import dataset. Using placeholder.")
            # Create placeholder datasets - you'll need to replace this
            self.train_dataset = None
            self.test_dataset = None
            return

        # Apply data augmentation
        if self.config.get('USE_AUGMENTATION', True):
            self.train_dataset = AugmentedTrajectoryDataset(self.train_dataset, self.config)
        
        # Apply multimodal targets
        if self.config.get('USE_MULTIMODAL_TARGETS', True):
            self.train_dataset = MultiModalTargetDataset(self.train_dataset, self.config)
            self.test_dataset = MultiModalTargetDataset(self.test_dataset, self.config)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
        )
        self.test_data
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

        # For predictions
        predict_indices = np.arange(2100)
        predict_config = {
            "INDICES": predict_indices,
            "DATA_FILENAME": "test.npz",
            "INFERENCE": True,
        }
        self.predict_dataset = dataset.AllAgentsNormalizedDataset(predict_config)
        self.predict_dataloader = DataLoader(
            self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

    def setUpModel(self):
        self.model = EnhancedTrajectoryPredictor(self.config)
        self.model.to(self.config["DEVICE"])
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(self.model)

    def setUpOptimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config["LEARNING_RATE"],
            weight_decay=self.config.get("WEIGHT_DECAY", 1e-4)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            patience=self.config.get("SCHEDULER_PATIENCE", 5), 
            factor=self.config.get("SCHEDULER_FACTOR", 0.5),
            min_lr=self.config.get("MIN_LR", 1e-6)
        )

    def train(self):
        training_progress = {}
        best_eval_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["EPOCHS"]):
            print(f"Performing epoch {epoch}")
            self.epoch = epoch
            
            train_metrics = self.train_epoch()
            eval_metrics = self.eval_epoch()
            
            # Log metrics
            print(f"Epoch {epoch}:")
            print(f"  Train - Total: {train_metrics['total_loss']:.5f}, Pos: {train_metrics['pos_loss']:.5f}")
            print(f"  Eval  - Total: {eval_metrics['total_loss']:.5f}, Pos: {eval_metrics['pos_loss']:.5f}")
            
            training_progress[epoch] = {
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            }

            # Learning rate scheduling
            self.scheduler.step(eval_metrics['total_loss'])
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  Current Learning Rate: {current_lr}")

            # Early stopping
            if eval_metrics['total_loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['total_loss']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.get("EARLY_STOPPING_PATIENCE", 10):
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Analysis
            if self.config.get("ANALYZE") and epoch % 5 == 0:
                origspace_score = self.analyze(self.train_dataloader)
                print(f"Original space score (train): {origspace_score:.5f}")

        return training_progress

    def train_epoch(self):
        self.model.train()
        
        total_loss = 0
        pos_loss_sum = 0
        vel_loss_sum = 0
        acc_loss_sum = 0
        num_batches = len(self.train_dataloader)

        for batch, (X, Y, indices) in enumerate(self.train_dataloader):
            X = X.to(self.config["DEVICE"])
            Y = Y.to(self.config["DEVICE"])
            
            # Forward pass
            predictions = self.model(X)
            
            # Compute multi-output loss
            loss, loss_components = self.enhanced_trainer.compute_multi_loss(predictions, Y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            pos_loss_sum += loss_components['pos_loss'].item()
            vel_loss_sum += loss_components['vel_loss'] if isinstance(loss_components['vel_loss'], torch.Tensor) else loss_components['vel_loss']
            acc_loss_sum += loss_components['acc_loss'] if isinstance(loss_components['acc_loss'], torch.Tensor) else loss_components['acc_loss']

        # Return average losses
        return {
            'total_loss': total_loss / num_batches,
            'pos_loss': pos_loss_sum / num_batches,
            'vel_loss': vel_loss_sum / num_batches,
            'acc_loss': acc_loss_sum / num_batches,
        }

    def eval_epoch(self):
        self.model.eval()
        
        total_loss = 0
        pos_loss_sum = 0
        vel_loss_sum = 0
        acc_loss_sum = 0
        num_batches = len(self.test_dataloader)

        with torch.no_grad():
            for batch, (X, Y, _) in enumerate(self.test_dataloader):
                X = X.to(self.config["DEVICE"])
                Y = Y.to(self.config["DEVICE"])

                # Forward pass
                predictions = self.model(X)
                
                # Compute multi-output loss
                loss, loss_components = self.enhanced_trainer.compute_multi_loss(predictions, Y)
                
                # Accumulate losses
                total_loss += loss.item()
                pos_loss_sum += loss_components['pos_loss'].item()
                vel_loss_sum += loss_components['vel_loss'] if isinstance(loss_components['vel_loss'], torch.Tensor) else loss_components['vel_loss']
                acc_loss_sum += loss_components['acc_loss'] if isinstance(loss_components['acc_loss'], torch.Tensor) else loss_components['acc_loss']

        # Return average losses
        return {
            'total_loss': total_loss / num_batches,
            'pos_loss': pos_loss_sum / num_batches,
            'vel_loss': vel_loss_sum / num_batches,
            'acc_loss': acc_loss_sum / num_batches,
        }

    def analyze(self, dataloader):
        """Analyze model performance in original space"""
        num_batches = self.config.get("ANALYZE_NUM_BATCHES", 50)
        self.model.eval()
        
        total_unnormalized = 0
        position_errors = []
        
        with torch.no_grad():
            for batch, (X, Y, org_indices) in enumerate(dataloader):
                if batch >= num_batches:
                    break

                X = X.to(self.config["DEVICE"])
                Y = Y.to(self.config["DEVICE"])

                # Get predictions
                predictions = self.model(X)
                position_pred = predictions['position']  # Focus on position for evaluation

                # Convert to original space (assuming your dataset has these methods)
                try:
                    true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                        Y.cpu().detach().numpy(), org_indices, indicator="true"
                    )
                    pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                        position_pred.cpu().detach().numpy(), org_indices, indicator="prediction"
                    )
                    
                    # Compute original space metric
                    unnormalized_metric = dataloader.dataset.computeOriginalSpaceMetric(
                        true_unnormalized, pred_unnormalized
                    )
                    total_unnormalized += unnormalized_metric
                    
                    # Compute additional metrics
                    batch_errors = np.mean(np.sqrt(np.sum((true_unnormalized - pred_unnormalized)**2, axis=-1)))
                    position_errors.append(batch_errors)
                    
                except AttributeError:
                    # Fallback if dataset doesn't have these methods
                    mse = torch.mean((Y - position_pred) ** 2).item()
                    total_unnormalized += mse
                    position_errors.append(np.sqrt(mse))

        avg_score = total_unnormalized / min(num_batches, len(dataloader))
        avg_position_error = np.mean(position_errors)
        
        print(f"  Average position error: {avg_position_error:.5f}")
        return avg_score

    def predict(self):
        """Generate predictions for test set"""
        all_predictions = []
        self.model.eval()

        with torch.no_grad():
            for batch, (X, _, org_indices) in enumerate(self.predict_dataloader):
                X = X.to(self.config["DEVICE"])
                
                # Get predictions
                predictions = self.model(X)
                position_pred = predictions['position']  # Shape: (batch_size, 60, 2)
                
                # Convert to numpy
                pred_numpy = position_pred.cpu().detach().numpy()
                
                # Unnormalize if method exists
                try:
                    pred_unnormalized = self.predict_dataloader.dataset.unnormalizeData(
                        pred_numpy, org_indices
                    )
                    all_predictions.append(pred_unnormalized)
                except AttributeError:
                    # Fallback - use predictions as is
                    all_predictions.append(pred_numpy)

        # Concatenate all predictions
        all_np_predictions = np.concatenate(all_predictions, axis=0)
        
        # Ensure correct shape: (2100, 60, 2)
        if all_np_predictions.shape != (2100, 60, 2):
            print(f"Warning: Prediction shape is {all_np_predictions.shape}, expected (2100, 60, 2)")
            all_np_predictions = all_np_predictions[:2100, :60, :2]  # Clip if necessary
        
        self.convertAndSavePredictions(all_np_predictions)
        return all_np_predictions

    def convertAndSavePredictions(self, predictions):
        """Convert predictions to submission format"""
        assert predictions.shape == (2100, 60, 2), f"Expected shape (2100, 60, 2), got {predictions.shape}"

        # Reshape to (2100 * 60, 2)
        pred_output = predictions.reshape(-1, 2)
        
        # Create DataFrame
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])
        output_df.index.name = "index"
        
        # Save to CSV
        output_path = os.path.join(utils.SUBMISSION_DIR, "enhanced_trajectory_submission.csv")
        output_df.to_csv(output_path)
        print(f"Predictions saved to {output_path}")

    def save_model_checkpoint(self, epoch, metrics):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_model_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config["DEVICE"])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']


def main():
    """Main function to run enhanced training"""
    
    # Get enhanced configuration
    config = get_enhanced_config()
    
    # Override with specific settings if needed
    config.update({
        "BATCH_SIZE": 16,  # Reduced due to increased model complexity
        "LEARNING_RATE": 0.0005,
        "EPOCHS": 80,
        "WEIGHT_DECAY": 1e-4,
        "EARLY_STOPPING_PATIENCE": 15,
        "SCHEDULER_PATIENCE": 7,
        "SCHEDULER_FACTOR": 0.5,
        "MIN_LR": 1e-6,
        
        # Enhanced model specific
        "SLIDING_WINDOW_SIZE": 8,
        "GNN_NUM_LAYERS": 2,  # Reduced complexity
        "TEMPORAL_NUM_LAYERS": 3,  # Reduced complexity
        
        # Loss weights (tune these based on validation performance)
        "POS_WEIGHT": 1.0,
        "VEL_WEIGHT": 0.05,
        "ACC_WEIGHT": 0.01,
    })

    print("Starting Enhanced Trajectory Prediction Training")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    # Create trainer and start training
    trainer = EnhancedAttentionAndNNTrainer(config)
    
    try:
        # Run the full pipeline
        training_progress = trainer.performPipeline()
        
        # Generate predictions
        print("\nGenerating predictions...")
        predictions = trainer.predict()
        
        print("\nTraining completed successfully!")
        print(f"Best model saved as 'best_model.pth'")
        
        return training_progress, predictions
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_inference_only(model_path, config):
    """Run inference with a trained model"""
    
    # Create trainer
    trainer = EnhancedAttentionAndNNTrainer(config)
    trainer.setUpData()
    trainer.setUpModel()
    
    # Load trained model
    trainer.model.load_state_dict(torch.load(model_path, map_location=config["DEVICE"]))
    print(f"Loaded model from {model_path}")
    
    # Generate predictions
    predictions = trainer.predict()
    return predictions


def hyperparameter_search():
    """Simple hyperparameter search"""
    
    base_config = get_enhanced_config()
    
    # Define search space
    search_configs = [
        {"LEARNING_RATE": 0.001, "GNN_NUM_LAYERS": 2, "TEMPORAL_NUM_LAYERS": 3},
        {"LEARNING_RATE": 0.0005, "GNN_NUM_LAYERS": 3, "TEMPORAL_NUM_LAYERS": 4},
        {"LEARNING_RATE": 0.0001, "GNN_NUM_LAYERS": 2, "TEMPORAL_NUM_LAYERS": 2},
    ]
    
    best_score = float('inf')
    best_config = None
    
    for i, search_params in enumerate(search_configs):
        print(f"\n{'='*20} Search {i+1}/{len(search_configs)} {'='*20}")
        
        # Create config for this search
        config = base_config.copy()
        config.update(search_params)
        config["EPOCHS"] = 20  # Reduced for search
        
        # Train model
        trainer = EnhancedAttentionAndNNTrainer(config)
        training_progress = trainer.performPipeline()
        
        # Get final validation score
        final_score = list(training_progress.values())[-1]['eval_metrics']['total_loss']
        
        print(f"Final validation score: {final_score:.5f}")
        
        if final_score < best_score:
            best_score = final_score
            best_config = config.copy()
    
    print(f"\nBest configuration found:")
    print(f"Best score: {best_score:.5f}")
    for key, value in best_config.items():
        if key in search_configs[0]:
            print(f"  {key}: {value}")
    
    return best_config



class AugmentedTrajectoryDataset(Dataset):
    """
    Dataset wrapper that applies data augmentation techniques for trajectory prediction
    """
    
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config
        self.augment_prob = config.get('AUGMENT_PROB', 0.5)
        
        # Augmentation parameters
        self.noise_std = config.get('NOISE_STD', 0.01)
        self.rotation_range = config.get('ROTATION_RANGE', 0.1)  # radians
        self.scale_range = config.get('SCALE_RANGE', (0.95, 1.05))
        self.time_shift_range = config.get('TIME_SHIFT_RANGE', 3)  # time steps
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base data
        X, Y, indices = self.base_dataset[idx]
        
        # Apply augmentation with probability
        if random.random() < self.augment_prob and self.base_dataset.training:
            X, Y = self.apply_augmentation(X, Y)
        
        return X, Y, indices
    
    def apply_augmentation(self, X, Y):
        """Apply various augmentation techniques"""
        
        # Choose augmentation techniques randomly
        augmentations = []
        
        if random.random() < 0.3:
            augmentations.append('noise')
        if random.random() < 0.2:
            augmentations.append('rotation')
        if random.random() < 0.2:
            augmentations.append('scale')
        if random.random() < 0.1:
            augmentations.append('time_shift')
        
        # Apply augmentations
        for aug in augmentations:
            if aug == 'noise':
                X, Y = self.add_noise(X, Y)
            elif aug == 'rotation':
                X, Y = self.apply_rotation(X, Y)
            elif aug == 'scale':
                X, Y = self.apply_scaling(X, Y)
            elif aug == 'time_shift':
                X = self.apply_time_shift(X)
        
        return X, Y
    
    def add_noise(self, X, Y):
        """Add Gaussian noise to positions and velocities"""
        noise_X = torch.randn_like(X[:, :, :4]) * self.noise_std
        noise_Y = torch.randn_like(Y) * self.noise_std
        
        X_aug = X.clone()
        X_aug[:, :, :4] += noise_X  # Add noise to position and velocity
        
        Y_aug = Y.clone()
        Y_aug += noise_Y
        
        return X_aug, Y_aug
    
    def apply_rotation(self, X, Y):
        """Apply random rotation to the scene"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix
        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=X.dtype)
        
        X_aug = X.clone()
        Y_aug = Y.clone()
        
        # Rotate positions
        pos = X_aug[:, :, :2]  # (num_agents, seq_len, 2)
        pos_flat = pos.reshape(-1, 2)  # (num_agents * seq_len, 2)
        pos_rot = torch.matmul(pos_flat, R.T)
        X_aug[:, :, :2] = pos_rot.reshape(pos.shape)
        
        # Rotate velocities
        vel = X_aug[:, :, 2:4]
        vel_flat = vel.reshape(-1, 2)
        vel_rot = torch.matmul(vel_flat, R.T)
        X_aug[:, :, 2:4] = vel_rot.reshape(vel.shape)
        
        # Rotate target positions
        Y_flat = Y_aug.reshape(-1, 2)
        Y_rot = torch.matmul(Y_flat, R.T)
        Y_aug = Y_rot.reshape(Y_aug.shape)
        
        # Update heading
        X_aug[:, :, 4] += angle
        
        return X_aug, Y_aug
    
    def apply_scaling(self, X, Y):
        """Apply random scaling to the scene"""
        scale = random.uniform(*self.scale_range)
        
        X_aug = X.clone()
        Y_aug = Y.clone()
        
        # Scale positions and velocities
        X_aug[:, :, :4] *= scale
        Y_aug *= scale
        
        return X_aug, Y_aug
    
    def apply_time_shift(self, X):
        """Apply random time shift to create temporal variation"""
        if X.shape[1] <= self.time_shift_range * 2:
            return X  # Skip if sequence too short
        
        shift = random.randint(-self.time_shift_range, self.time_shift_range)
        
        if shift > 0:
            # Shift forward: pad beginning with first frame
            first_frame = X[:, :1, :].expand(-1, shift, -1)
            X_shifted = torch.cat([first_frame, X[:, :-shift, :]], dim=1)
        elif shift < 0:
            # Shift backward: pad end with last frame
            last_frame = X[:, -1:, :].expand(-1, -shift, -1)
            X_shifted = torch.cat([X[:, -shift:, :], last_frame], dim=1)
        else:
            X_shifted = X
        
        return X_shifted


class SlidingWindowDataset(Dataset):
    """
    Dataset that creates multiple samples from each trajectory using sliding window
    """
    
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.window_size = config.get('SLIDING_WINDOW_SIZE', 30)
        self.stride = config.get('SLIDING_STRIDE', 10)
        
        # Pre-compute all valid windows
        self.windows = self._compute_windows()
    
    def _compute_windows(self):
        """Pre-compute all valid sliding windows"""
        windows = []
        
        for idx in range(len(self.base_dataset)):
            X, Y, indices = self.base_dataset[idx]
            seq_len = X.shape[1]
            
            # Create sliding windows
            for start in range(0, seq_len - self.window_size + 1, self.stride):
                end = start + self.window_size
                windows.append((idx, start, end))
        
        return windows
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        original_idx, start, end = self.windows[idx]
        X, Y, indices = self.base_dataset[original_idx]
        
        # Extract window
        X_window = X[:, start:end, :]
        
        # Adjust target if needed (this depends on your task)
        return X_window, Y, indices


class BalancedScenarioSampler:
    """
    Sampler that ensures balanced representation of different driving scenarios
    """
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        
        # Analyze scenarios in the dataset
        self.scenario_indices = self._analyze_scenarios()
    
    def _analyze_scenarios(self):
        """Analyze and categorize scenarios"""
        scenario_indices = {
            'straight': [],
            'turning': [],
            'stopping': [],
            'accelerating': [],
            'complex': []
        }
        
        print("Analyzing scenarios...")
        for idx in range(min(1000, len(self.dataset))):  # Sample for analysis
            try:
                X, Y, _ = self.dataset[idx]
                scenario = self._classify_scenario(X, Y)
                scenario_indices[scenario].append(idx)
            except:
                continue
        
        # Print statistics
        for scenario, indices in scenario_indices.items():
            print(f"{scenario}: {len(indices)} samples")
        
        return scenario_indices
    
    def _classify_scenario(self, X, Y):
        """Classify the driving scenario based on ego vehicle behavior"""
        ego_pos = X[0, :, :2]  # Ego vehicle positions
        ego_vel = X[0, :, 2:4]  # Ego vehicle velocities
        
        # Calculate metrics
        speed = torch.norm(ego_vel, dim=1)
        acceleration = torch.diff(speed)
        
        # Angular velocity (turning)
        pos_diff = torch.diff(ego_pos, dim=0)
        angles = torch.atan2(pos_diff[:, 1], pos_diff[:, 0])
        angular_vel = torch.diff(angles)
        
        # Classify based on behavior
        if torch.mean(torch.abs(angular_vel)) > 0.1:
            return 'turning'
        elif torch.mean(speed) < 0.5:
            return 'stopping'
        elif torch.mean(acceleration) > 0.2:
            return 'accelerating'
        elif torch.std(speed) < 0.1 and torch.std(angular_vel) < 0.05:
            return 'straight'
        else:
            return 'complex'
    
    def get_balanced_indices(self, num_samples):
        """Get balanced sample indices"""
        samples_per_scenario = num_samples // len(self.scenario_indices)
        balanced_indices = []
        
        for scenario, indices in self.scenario_indices.items():
            if len(indices) >= samples_per_scenario:
                sampled = random.sample(indices, samples_per_scenario)
            else:
                # If not enough samples, use all and repeat
                sampled = indices * (samples_per_scenario // len(indices) + 1)
                sampled = sampled[:samples_per_scenario]
            
            balanced_indices.extend(sampled)
        
        random.shuffle(balanced_indices)
        return balanced_indices


class MultiModalTargetDataset(Dataset):
    """
    Dataset that provides multiple prediction targets (position, velocity, acceleration)
    """
    
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        X, Y, indices = self.base_dataset[idx]
        
        # Create multi-modal targets
        targets = self._create_multimodal_targets(Y)
        
        return X, targets, indices
    
    def _create_multimodal_targets(self, Y):
        """Create position, velocity, and acceleration targets"""
        # Y is position trajectory (seq_len, 2)
        
        # Position target (already provided)
        pos_target = Y
        
        # Velocity target (derivative of position)
        vel_target = torch.zeros_like(Y)
        vel_target[1:, :] = Y[1:, :] - Y[:-1, :]
        
        # Acceleration target (derivative of velocity)
        acc_target = torch.zeros_like(Y)
        acc_target[1:, :] = vel_target[1:, :] - vel_target[:-1, :]
        
        return {
            'position': pos_target,
            'velocity': vel_target,
            'acceleration': acc_target
        }


def create_augmented_datasets(train_dataset, test_dataset, config):
    """
    Create augmented training and test datasets with various enhancements
    """
    
    # Wrap datasets with augmentation
    augmented_train = AugmentedTrajectoryDataset(train_dataset, config)
    
    # Optionally create sliding window dataset for more training data
    if config.get('USE_SLIDING_WINDOW', False):
        sliding_train = SlidingWindowDataset(augmented_train, config)
        print(f"Created sliding window dataset with {len(sliding_train)} samples")
        augmented_train = sliding_train
    
    # Create multi-modal targets if needed
    if config.get('USE_MULTIMODAL_TARGETS', True):
        multimodal_train = MultiModalTargetDataset(augmented_train, config)
        multimodal_test = MultiModalTargetDataset(test_dataset, config)
        print("Created multi-modal target datasets")
        return multimodal_train, multimodal_test
    
    return augmented_train, test_dataset


def analyze_dataset_statistics(dataset, num_samples=1000):
    """Analyze and print dataset statistics"""
    
    print("Analyzing dataset statistics...")
    
    speeds = []
    accelerations = []
    turning_rates = []
    
    for idx in range(min(num_samples, len(dataset))):
        try:
            X, Y, _ = dataset[idx]
            
            # Analyze ego vehicle (agent 0)
            ego_vel = X[0, :, 2:4]  # Velocities
            ego_pos = X[0, :, :2]   # Positions
            
            # Speed
            speed = torch.norm(ego_vel, dim=1)
            speeds.extend(speed.tolist())
            
            # Acceleration
            if len(speed) > 1:
                acc = torch.diff(speed)
                accelerations.extend(acc.tolist())
            
            # Turning rate
            if len(ego_pos) > 2:
                pos_diff = torch.diff(ego_pos, dim=0)
                angles = torch.atan2(pos_diff[:, 1], pos_diff[:, 0])
                if len(angles) > 1:
                    angular_vel = torch.diff(angles)
                    turning_rates.extend(angular_vel.tolist())
                    
        except Exception as e:
            continue
    
    # Print statistics
    print(f"Speed statistics:")
    print(f"  Mean: {np.mean(speeds):.3f}, Std: {np.std(speeds):.3f}")
    print(f"  Min: {np.min(speeds):.3f}, Max: {np.max(speeds):.3f}")
    
    print(f"Acceleration statistics:")
    print(f"  Mean: {np.mean(accelerations):.3f}, Std: {np.std(accelerations):.3f}")
    print(f"  Min: {np.min(accelerations):.3f}, Max: {np.max(accelerations):.3f}")
    
    print(f"Turning rate statistics:")
    print(f"  Mean: {np.mean(turning_rates):.3f}, Std: {np.std(turning_rates):.3f}")
    print(f"  Min: {np.min(turning_rates):.3f}, Max: {np.max(turning_rates):.3f}")

def get_enhanced_config():
    return {
        "RAW_FEATURES": 9,  # Matches your input
        "AUGMENTED_FEATURES": 15,  # After FeatureAugmenter
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
# Example usage configuration
def get_augmentation_config():
    return {
        # Data augmentation
        'AUGMENT_PROB': 0.7,
        'NOISE_STD': 0.02,
        'ROTATION_RANGE': 0.15,
        'SCALE_RANGE': (0.9, 1.1),
        'TIME_SHIFT_RANGE': 5,
        
        # Sliding window
        'USE_SLIDING_WINDOW': False,  # Can increase training data significantly
        'SLIDING_WINDOW_SIZE': 30,
        'SLIDING_STRIDE': 10,
        
        # Multi-modal targets
        'USE_MULTIMODAL_TARGETS': True,
        
        # Balanced sampling
        'USE_BALANCED_SAMPLING': False,  # Enable if dataset is imbalanced
    }