import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import utilities as utils
from sklearn.model_selection import train_test_split
import LSTMWeightBasedLoss.data as dataset
from NewTransformer.compatible_enhanced_model import EgoAgentEnsembleModel

from NewTransformer.compatible_enhanced_model import EnhancedEgoAgentNN

class EnhancedEgoAgentDataset:
    """Enhanced dataset with data augmentation and multi-agent features"""
    def __init__(self, config):
        self.config = config
        # Load data using your existing structure
        if config.get("INFERENCE", False):
            # For inference, load test data
            data_file = np.load(config["DATA_FILENAME"])
            self.input_data = data_file['input']  # Shape: (2100, 50, 50, 6)
            self.target_data = None
        else:
            # For training, load train/val data  
            data_file = np.load(config["DATA_FILENAME"])
            self.input_data = data_file['input']   # Shape: (10000, 50, 50, 6) 
            self.target_data = data_file['target'] # Shape: (10000, 60, 5)
            
        self.indices = config["INDICES"]
        self.inference = config.get("INFERENCE", False)
        
        # Load normalization parameters if they exist
        self.load_normalization_params()
        
    def load_normalization_params(self):
        """Load pre-computed normalization parameters"""
        # These should be computed from training data
        self.pos_mean = np.array([0.0, 0.0])
        self.pos_std = np.array([50.0, 50.0])
        self.vel_mean = np.array([0.0, 0.0])
        self.vel_std = np.array([10.0, 10.0])
        self.heading_std = np.pi
        
    def compute_acceleration(self, positions, velocities):
        """Compute acceleration from position and velocity data"""
        # Acceleration from velocity differences
        vel_diff = np.diff(velocities, axis=1, prepend=velocities[:, :1])
        acc = vel_diff  # dt = 0.1s, so acceleration is velocity difference * 10
        
        # Alternative: acceleration from position differences
        pos_diff = np.diff(positions, axis=1, prepend=positions[:, :1])
        pos_acc = np.diff(pos_diff, axis=1, prepend=pos_diff[:, :1])
        
        return acc, pos_acc
    
    def create_sliding_windows(self, data, window_size=10):
        """Create sliding windows for better temporal modeling"""
        batch_size, seq_len, num_features = data.shape
        if seq_len < window_size:
            return data
            
        windowed_data = []
        for i in range(seq_len - window_size + 1):
            window = data[:, i:i+window_size, :]
            windowed_data.append(window)
        
        return np.stack(windowed_data, axis=1)  # (batch, num_windows, window_size, features)
    
    def extract_multi_agent_features(self, scene_data):
        """Extract features from all agents while handling zero-padding"""
        batch_size, num_agents, seq_len, num_features = scene_data.shape
        
        # Separate ego agent (index 0) from other agents
        ego_data = scene_data[:, 0, :, :]  # (batch, seq_len, features)
        other_agents = scene_data[:, 1:, :, :]  # (batch, num_agents-1, seq_len, features)
        
        # Create mask for valid agents (non-zero padded)
        agent_mask = np.any(other_agents != 0, axis=(2, 3))  # (batch, num_agents-1)
        
        # Compute relative features to ego agent
        ego_pos = ego_data[:, :, :2]  # (batch, seq_len, 2)
        ego_pos_expanded = ego_pos[:, None, :, :]  # (batch, 1, seq_len, 2)
        
        other_pos = other_agents[:, :, :, :2]  # (batch, num_agents-1, seq_len, 2)
        relative_pos = other_pos - ego_pos_expanded  # Relative positions
        
        # Distance and angle to ego
        distances = np.linalg.norm(relative_pos, axis=-1)  # (batch, num_agents-1, seq_len)
        angles = np.arctan2(relative_pos[:, :, :, 1], relative_pos[:, :, :, 0])
        
        # Aggregate features across other agents (weighted by distance)
        weights = 1.0 / (distances + 1e-6)  # Inverse distance weighting
        weights = weights * agent_mask[:, :, None]  # Apply agent mask
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-6)  # Normalize
        
        # Weighted aggregation of other agents' features
        other_vel = other_agents[:, :, :, 2:4]  # (batch, num_agents-1, seq_len, 2)
        other_heading = other_agents[:, :, :, 4:5]  # (batch, num_agents-1, seq_len, 1)
        
        # Aggregate features
        agg_rel_pos = np.sum(relative_pos * weights[:, :, :, None], axis=1)  # (batch, seq_len, 2)
        agg_rel_vel = np.sum(other_vel * weights[:, :, :, None], axis=1)  # (batch, seq_len, 2)
        agg_distances = np.sum(distances * weights, axis=1)  # (batch, seq_len)
        
        # Count of nearby agents
        nearby_count = np.sum((distances < 20.0) * agent_mask[:, :, None], axis=1)  # (batch, seq_len)
        
        return agg_rel_pos, agg_rel_vel, agg_distances[:, :, None], nearby_count[:, :, None]
    
    def augment_features(self, ego_data):
        """Add acceleration and other derived features"""
        positions = ego_data[:, :, :2]
        velocities = ego_data[:, :, 2:4]
        heading = ego_data[:, :, 4:5]
        
        # Compute accelerations
        vel_acc, pos_acc = self.compute_acceleration(positions, velocities)
        
        # Speed (magnitude of velocity)
        speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
        
        # Angular velocity (change in heading)
        angular_vel = np.diff(heading, axis=1, prepend=heading[:, :1])
        
        # Curvature approximation
        vel_mag = np.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-6
        curvature = np.abs(angular_vel) / vel_mag
        
        return vel_acc, pos_acc, speed, angular_vel, curvature
    
    def normalize_features(self, features):
        """Normalize features for better training stability"""
        # This is a simplified normalization - in practice, compute from training data
        normalized = features.copy()
        
        # Only normalize if we have the expected 18 features
        if features.shape[-1] >= 5:
            # Normalize positions (first 2 features)
            normalized[:, :2] = (features[:, :2] - self.pos_mean) / self.pos_std
            
            # Normalize velocities (features 2-3)
            normalized[:, 2:4] = (features[:, 2:4] - self.vel_mean) / self.vel_std
            
            # Normalize heading (feature 4)
            normalized[:, 4] = features[:, 4] / self.heading_std
            
            # For additional features (5-17), use simple standardization
            if features.shape[-1] > 5:
                for i in range(5, features.shape[-1]):
                    feat_std = np.std(features[:, i]) + 1e-6
                    feat_mean = np.mean(features[:, i])
                    normalized[:, i] = (features[:, i] - feat_mean) / feat_std
        
        return normalized
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        scene_idx = self.indices[idx]
        
        if self.inference:
            # For inference: input shape is (50_agents, 50_timesteps, 6_features)
            scene_data = self.input_data[scene_idx]  # (50, 50, 6)
            
            # Extract ego agent data (agent 0)
            ego_data = scene_data[0, :, :5]  # (50, 5) - exclude object_type
            
            # Extract multi-agent features  
            # Need to reshape for multi-agent processing: (1, 50, 50, 5)
            multi_agent_scene = scene_data[None, :, :, :5]  # Add batch dim
            agg_rel_pos, agg_rel_vel, agg_distances, nearby_count = self.extract_multi_agent_features(multi_agent_scene)
            
            # Squeeze batch dimension
            agg_rel_pos = agg_rel_pos[0]  # (50, 2)
            agg_rel_vel = agg_rel_vel[0]  # (50, 2)
            agg_distances = agg_distances[0]  # (50, 1)
            nearby_count = nearby_count[0]  # (50, 1)
            
            # Augment ego features
            vel_acc, pos_acc, speed, angular_vel, curvature = self.augment_features(ego_data[None, :, :])
            vel_acc, pos_acc, speed, angular_vel, curvature = vel_acc[0], pos_acc[0], speed[0], angular_vel[0], curvature[0]
            
            # Combine all features
            combined_features = np.concatenate([
                ego_data,  # Original 5 features (50, 5)
                vel_acc, pos_acc,  # Acceleration features (50, 4)
                speed, angular_vel, curvature,  # Derived features (50, 3)
                agg_rel_pos, agg_rel_vel,  # Multi-agent features (50, 4)
                agg_distances, nearby_count  # Spatial features (50, 2)
            ], axis=-1)  # Total: (50, 18)
            
            # Normalize
            combined_features = self.normalize_features(combined_features)
            
            return torch.FloatTensor(combined_features), torch.zeros(60, 5), scene_idx
        
        else:
            # Training mode: input shape is (50_agents, 50_timesteps, 6_features)
            scene_data = self.input_data[scene_idx]  # (50, 50, 6)
            target_data = self.target_data[scene_idx]  # (60, 5) - ego agent future trajectory
            
            # Extract ego agent data (agent 0)
            ego_data = scene_data[0, :, :5]  # (50, 5) - exclude object_type
            
            # Extract multi-agent features
            multi_agent_scene = scene_data[None, :, :, :5]  # Add batch dim
            agg_rel_pos, agg_rel_vel, agg_distances, nearby_count = self.extract_multi_agent_features(multi_agent_scene)
            
            # Squeeze batch dimension
            agg_rel_pos = agg_rel_pos[0]  # (50, 2)
            agg_rel_vel = agg_rel_vel[0]  # (50, 2)
            agg_distances = agg_distances[0]  # (50, 1)
            nearby_count = nearby_count[0]  # (50, 1)
            
            # Augment ego features
            vel_acc, pos_acc, speed, angular_vel, curvature = self.augment_features(ego_data[None, :, :])
            vel_acc, pos_acc, speed, angular_vel, curvature = vel_acc[0], pos_acc[0], speed[0], angular_vel[0], curvature[0]
            
            # Combine all features
            combined_features = np.concatenate([
                ego_data,  # Original 5 features (50, 5)
                vel_acc, pos_acc,  # Acceleration features (50, 4)
                speed, angular_vel, curvature,  # Derived features (50, 3)
                agg_rel_pos, agg_rel_vel,  # Multi-agent features (50, 4)
                agg_distances, nearby_count  # Spatial features (50, 2)
            ], axis=-1)  # Total: (50, 18)
            
            # Normalize
            combined_features = self.normalize_features(combined_features)
            
            return torch.FloatTensor(combined_features), torch.FloatTensor(target_data), scene_idx


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for agent interactions"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output


class EnhancedEgoAgentModel(nn.Module):
    """Enhanced model with multi-agent awareness and attention mechanism"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.input_size = 18  # Enhanced feature size
        self.hidden_size = 256
        self.num_layers = 3
        self.output_steps = 60
        self.output_size = 5  # [pos_x, pos_y, vel_x, vel_y, heading]
        
        # Feature encoding layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(128, 128)
        )
        
        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config["DROPOUT"] if self.num_layers > 1 else 0
        )
        
        # Self-attention for temporal dependencies
        self.temporal_attention = MultiHeadAttention(self.hidden_size, num_heads=8, dropout=config["DROPOUT"])
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(256, self.output_steps * self.output_size)
        ])
        
        # Skip connection
        self.skip_connection = nn.Linear(self.hidden_size, self.output_steps * self.output_size)
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(self.output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Feature encoding
        x_encoded = self.feature_encoder(x)  # (batch, seq, 128)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_encoded)  # (batch, seq, hidden_size)
        
        # Temporal attention
        attended = self.temporal_attention(lstm_out, lstm_out, lstm_out)  # (batch, seq, hidden_size)
        
        # Use final time step for prediction
        final_state = attended[:, -1, :]  # (batch, hidden_size)
        
        # Decoder with skip connection
        x_dec = final_state
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Linear):
                x_dec = layer(x_dec)
            else:
                x_dec = layer(x_dec)
        
        # Skip connection
        skip = self.skip_connection(final_state)
        output = x_dec + skip  # Residual connection
        
        # Reshape and normalize
        output = output.view(batch_size, self.output_steps, self.output_size)
        
        # Apply layer normalization to each time step
        output = self.output_norm(output)
        
        return output


class EnhancedTrainer:
    """Enhanced trainer with better loss function and evaluation"""
    def __init__(self, config):
        self.config = config
        self.best_eval_loss = float('inf')
        
    def setup_data(self):
        """Setup enhanced datasets"""
        total_indices = np.arange(self.config["NUM_SAMPLES"])
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )
        
        train_config = {
            "INDICES": train_indices,
            "DATA_FILENAME": "train.npz",
        }
        test_config = {
            "INDICES": test_indices,
            "DATA_FILENAME": "train.npz"
        }
        
        self.train_dataset = dataset.EgoAgentNormalizedDataset(train_config)
        self.test_dataset = dataset.EgoAgentNormalizedDataset(test_config)
        
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
        )
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

        self.predict_dataset = dataset.EgoAgentNormalizedDataset(predict_config)

        self.predict_dataloader = DataLoader(
            self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )
    
    def setup_model(self):
        """Setup enhanced model"""
        #self.model = EnhancedEgoAgentModel(self.config).to(self.config["DEVICE"])
        self.model = EgoAgentEnsembleModel(self.config).to(self.config["DEVICE"])
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_optimizer(self):
        """Setup optimizer with warmup and scheduling"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config["LEARNING_RATE"],
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["LEARNING_RATE"],
            epochs=self.config["EPOCHS"],
            steps_per_epoch=len(self.train_dataloader),
            pct_start=0.1
        )
    
    def enhanced_loss(self, pred, target, weights=None):
        """Enhanced loss function with temporal and feature weighting"""
        if weights is None:
            # Position gets higher weight, especially for later time steps
            pos_weight = torch.linspace(1.0, 3.0, 60).to(pred.device)  # Increasing weight over time
            vel_weight = torch.ones(60).to(pred.device) * 0.5
            heading_weight = torch.ones(60).to(pred.device) * 0.3
            
            temporal_weights = torch.stack([
                pos_weight, pos_weight,  # x, y positions
                vel_weight, vel_weight,  # x, y velocities  
                heading_weight  # heading
            ], dim=1)  # (60, 5)
        else:
            temporal_weights = weights
            
        # MSE with temporal weighting
        mse_loss = ((pred - target) ** 2) * temporal_weights[None, :, :]
        
        # L1 loss for robustness
        l1_loss = torch.abs(pred - target) * temporal_weights[None, :, :] * 0.1
        
        # Total loss
        total_loss = mse_loss.mean() + l1_loss.mean()
        
        return total_loss
    
    def train_epoch(self):
        """Enhanced training epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, (X, y, _) in enumerate(self.train_dataloader):
            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
            
            # Forward pass
            predictions = self.model(X)
            
            # Compute loss
            loss = self.enhanced_loss(predictions, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / num_batches
    
    def eval_epoch(self):
        """Enhanced evaluation epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.test_dataloader)
        
        with torch.no_grad():
            for X, y, _ in self.test_dataloader:
                X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
                
                predictions = self.model(X)
                loss = self.enhanced_loss(predictions, y)
                
                total_loss += loss.item()
                
        return total_loss / num_batches
    
    def train(self):
        """Main training loop"""
        training_progress = {}
        
        for epoch in range(self.config["EPOCHS"]):
            train_loss = self.train_epoch()
            eval_loss = self.eval_epoch()
            
            print(f"Epoch {epoch+1}/{self.config['EPOCHS']}: "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {eval_loss:.6f}")
            
            training_progress[epoch] = {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }
            
            # Save best model
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with val loss: {eval_loss:.6f}")
        
        return training_progress
    
    def predict(self):
        """Generate predictions for test set"""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for X, _, org_indices in self.predict_dataloader:
                X = X.to(self.config["DEVICE"])
                
                predictions = self.model(X)
                
                # Convert to numpy and focus on position (first 2 dimensions)
                pred_np = predictions.cpu().numpy()[:, :, :2]  # Only x, y positions
                all_predictions.append(pred_np)
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Save predictions
        self.save_predictions(all_predictions)
        
        return all_predictions
    
    def save_predictions(self, predictions):
        """Save predictions to CSV format"""
        assert predictions.shape == (2100, 60, 2), f"Expected shape (2100, 60, 2), got {predictions.shape}"
        
        # Reshape to required format
        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])
        output_df.index.name = "index"
        
        # Save to CSV
        os.makedirs(utils.SUBMISSION_DIR, exist_ok=True)
        output_path = os.path.join(utils.SUBMISSION_DIR, "enhanced_trajectory_submission.csv")
        output_df.to_csv(output_path)
        print(f"Predictions saved to {output_path}")


def main():
    """Main training function"""
    config = {
        "BATCH_SIZE": 32,  # Reduced due to increased model complexity
        "LEARNING_RATE": 0.001,
        "EPOCHS": 50,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.2,
        "NUM_SAMPLES": 10000,
        "DROPOUT": 0.2,
    }
    
    print(f"Using device: {config['DEVICE']}")
    
    # Initialize trainer
    trainer = EnhancedTrainer(config)
    
    # Setup data, model, and optimizer
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_optimizer()
    
    # Train model
    print("Starting training...")
    training_progress = trainer.train()
    
    # Generate predictions
    print("Generating predictions...")
    predictions = trainer.predict()
    
    # Plot training progress
    df_progress = pd.DataFrame.from_dict(training_progress, orient="index")
    plt.figure(figsize=(10, 6))
    df_progress[["train_loss", "eval_loss"]].plot()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Enhanced Model Training Progress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("enhanced_training_progress.png")
    plt.show()
    
    print("Training completed!")

if __name__ == "__main__":
    main()