import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import utilities as utils
import AttentionAndNN.data as dataset

# Base class for all predictors
class BasePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config['D_INPUT']
        self.output_dim = config['D_OUTPUT']
        self.device = config['DEVICE']
        
    def forward(self, x):
        raise NotImplementedError

# 1. Transformer-based Predictor (Your existing model)
class TransformerPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        # ... (keep rest of init the same)
        
    def forward(self, inp):
        batch_size = inp.size(0)
        num_queries = self.config['NUM_QUERIES']
        dmodel = self.config['D_MODEL']
        
        # Process input - reshape to (batch, 50, 50, 9) based on data shape shown in logs
        inp_reshaped = inp.view(batch_size, 50, 50, 9)
        
        queries = self.queriesEmbedding(torch.arange(num_queries).to(self.device))
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        other_agent_inp = self.upscaleTransform(torch.flatten(inp_reshaped.transpose(1, 2), start_dim=2))
        precompact_inp = torch.cat((queries, other_agent_inp), dim=1)
        compact_inp = torch.flatten(self.encoder(precompact_inp)[:, :num_queries], start_dim=1)
        
        # Ego agent features (first 4 features: x, y, vx, vy)
        ego_inp = torch.flatten(inp_reshaped[:, 0, :, :4], start_dim=1)
        full_pred_inp = torch.cat((ego_inp, compact_inp), dim=1)
        
        output = self.prediction_nn(full_pred_inp)
        return output.view(-1, 60, 2)

# 2. LSTM-based Predictor
class LSTMPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        
        # Input processing
        self.input_projection = nn.Linear(self.input_dim, 256)
        
        # LSTM layers - increased num_layers to 2 to use dropout
        self.lstm1 = nn.LSTM(256, 512, batch_first=True, num_layers=2, dropout=config["DROPOUT"])
        self.lstm2 = nn.LSTM(512, 256, batch_first=True, num_layers=2, dropout=config["DROPOUT"])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(512, self.output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x_proj = self.input_projection(x)
        x_seq = x_proj.unsqueeze(1)  # Add sequence dimension
        
        # LSTM processing
        lstm_out1, _ = self.lstm1(x_seq)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Take last output
        final_features = lstm_out2[:, -1, :]
        
        # Generate predictions
        output = self.output_layers(final_features)
        return output.view(-1, 60, 2)

# 3. CNN-based Predictor
class CNNPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        
        # Reshape input to treat as 2D spatial data
        self.input_channels = 9  # Changed from 6 to match actual input channels
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to spatial format: (batch, 50, 110, 6) -> (batch, 6, 50, 110)
        x_reshaped = x.view(batch_size, 50, 110, 6).permute(0, 3, 1, 2)
        
        # CNN processing
        conv_features = self.conv_layers(x_reshaped)
        conv_features = conv_features.view(batch_size, -1)
        
        # FC processing
        output = self.fc_layers(conv_features)
        return output.view(-1, 60, 2)

# 4. Graph Neural Network Predictor
class GNNPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        
        self.node_features = 128
        self.edge_features = 64
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(9, self.node_features),  # Changed from 6 to 9 to match input
            nn.ReLU(),
            nn.Linear(self.node_features, self.node_features)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, self.edge_features),  # Distance and angle
            nn.ReLU(),
            nn.Linear(self.edge_features, self.edge_features)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(self.node_features, self.node_features, dropout=config["DROPOUT"])
            for _ in range(3)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(self.node_features, 512),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(512, self.output_dim)
        )
        
    def compute_edge_features(self, positions):
        """Compute edge features based on agent positions"""
        batch_size, num_agents = positions.shape[:2]
        
        # Compute pairwise distances and angles
        pos_expanded_i = positions.unsqueeze(2).expand(-1, -1, num_agents, -1)
        pos_expanded_j = positions.unsqueeze(1).expand(-1, num_agents, -1, -1)
        
        # Distance
        diff = pos_expanded_i - pos_expanded_j
        distances = torch.norm(diff, dim=-1, keepdim=True)
        
        # Angle
        angles = torch.atan2(diff[..., 1:2], diff[..., 0:1])
        
        edge_features = torch.cat([distances, angles], dim=-1)
        return edge_features
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input: (batch, 50*110*6) -> (batch, 50, 110, 6)
        x_reshaped = x.view(batch_size, 50, 110, 6)
        
        # Use last timestep for graph construction
        current_state = x_reshaped[:, :, -1, :]  # (batch, 50, 6)
        
        # Encode node features
        node_features = self.node_encoder(current_state)  # (batch, 50, node_features)
        
        # Compute edge features
        positions = current_state[:, :, :2]  # x, y positions
        edge_features = self.compute_edge_features(positions)
        edge_features = self.edge_encoder(edge_features)
        
        # Apply graph attention layers
        for gat_layer in self.gat_layers:
            node_features = gat_layer(node_features, edge_features)
        
        # Focus on ego agent (index 0)
        ego_features = node_features[:, 0, :]  # (batch, node_features)
        
        # Generate predictions
        output = self.output_layers(ego_features)
        return output.view(-1, 60, 2)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, node_features, edge_features):
        batch_size, num_nodes = node_features.shape[:2]
        
        # Linear transformation
        h = self.W(node_features)  # (batch, num_nodes, out_features)
        
        # Compute attention coefficients
        h_expanded_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        h_expanded_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        attention_input = torch.cat([h_expanded_i, h_expanded_j], dim=-1)
        e = self.leakyrelu(self.a(attention_input).squeeze(-1))
        
        # Mask self-attention and apply softmax
        mask = torch.eye(num_nodes).bool().to(node_features.device)
        e.masked_fill_(mask.unsqueeze(0), -1e9)
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention
        h_prime = torch.matmul(attention, h)
        
        return h_prime

# Ensemble Trainer
class ArchitectureEnsembleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['DEVICE']
        
        # Initialize all models
        self.models = {
            'transformer': TransformerPredictor(config),
            'lstm': LSTMPredictor(config),
            'cnn': CNNPredictor(config),
            'gnn': GNNPredictor(config)
        }
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
        
        # Initialize optimizers
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
            for name, model in self.models.items()
        }
        
        # Initialize schedulers
        self.schedulers = {
            name: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            for name, optimizer in self.optimizers.items()
        }
        
        # Validation performance tracking
        self.val_scores = {name: [] for name in self.models.keys()}
        self.ensemble_weights = {name: 0.25 for name in self.models.keys()}  # Equal weights initially
        
    def setup_data(self):
        """Setup data loaders"""
        total_indices = np.arange(self.config["NUM_SAMPLES"])
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )
        
        train_config = {"INDICES": train_indices, "DATA_FILENAME": "train.npz"}
        test_config = {"INDICES": test_indices, "DATA_FILENAME": "train.npz"}
        
        self.train_dataset = dataset.AllAgentsNormalizedDataset(train_config)
        self.test_dataset = dataset.AllAgentsNormalizedDataset(test_config)
        
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )
        
        # Prediction data
        predict_indices = np.arange(2100)
        predict_config = {"INDICES": predict_indices, "DATA_FILENAME": "test.npz", "INFERENCE": True}
        self.predict_dataset = dataset.AllAgentsNormalizedDataset(predict_config)
        self.predict_dataloader = DataLoader(
            self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )
    
    def compute_loss(self, predictions, targets):
        """Compute MSE loss with physics constraints"""
        mse_loss = F.mse_loss(predictions, targets)
        
        # Add physics constraints
        physics_loss = self.compute_physics_loss(predictions)
        
        return mse_loss + 0.1 * physics_loss
    
    def compute_physics_loss(self, predictions, dt=0.1):
        """Compute physics-based loss for realistic motion"""
        positions = predictions  # Shape: (batch, 60, 2)
        
        # Compute velocities and accelerations
        velocities = torch.diff(positions, dim=1) / dt
        accelerations = torch.diff(velocities, dim=1) / dt
        
        # Penalize unrealistic accelerations (> 8 m/s²)
        max_accel = 8.0
        accel_magnitude = torch.norm(accelerations, dim=2)
        accel_penalty = torch.relu(accel_magnitude - max_accel).mean()
        
        # Smoothness penalty (jerk)
        jerk = torch.diff(accelerations, dim=1) / dt
        smoothness_penalty = torch.norm(jerk, dim=2).mean()
        
        return accel_penalty + 0.1 * smoothness_penalty
    
    def train_single_model(self, model_name, model, optimizer, scheduler):
        """Train a single model"""
        print(f"\nTraining {model_name} model...")
        
        model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, (X, Y, _) in enumerate(self.train_dataloader):
            X, Y = X.to(self.device), Y.to(self.device)
            
            optimizer.zero_grad()
            predictions = model(X)
            loss = self.compute_loss(predictions, Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        
        # Validation
        val_loss = self.validate_single_model(model)
        scheduler.step(val_loss)
        
        self.val_scores[model_name].append(val_loss)
        
        print(f"  {model_name} - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        return avg_loss, val_loss
    
    def validate_single_model(self, model):
        """Validate a single model"""
        model.eval()
        total_loss = 0
        num_batches = len(self.test_dataloader)
        
        with torch.no_grad():
            for X, Y, _ in self.test_dataloader:
                X, Y = X.to(self.device), Y.to(self.device)
                predictions = model(X)
                loss = self.compute_loss(predictions, Y)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def update_ensemble_weights(self):
        """Update ensemble weights based on validation performance"""
        if all(len(scores) > 0 for scores in self.val_scores.values()):
            # Use inverse of recent validation loss as weight
            recent_scores = {name: np.mean(scores[-3:]) for name, scores in self.val_scores.items()}
            
            # Convert to weights (lower loss = higher weight)
            inv_scores = {name: 1.0 / (score + 1e-6) for name, score in recent_scores.items()}
            total_inv_score = sum(inv_scores.values())
            
            self.ensemble_weights = {name: score / total_inv_score for name, score in inv_scores.items()}
            
            print(f"\nUpdated ensemble weights: {self.ensemble_weights}")
    
    def train_ensemble(self):
        """Train all models in the ensemble"""
        print("Starting ensemble training...")
        self.setup_data()
        
        training_history = {name: {'train_loss': [], 'val_loss': []} for name in self.models.keys()}
        
        for epoch in range(self.config["EPOCHS"]):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config['EPOCHS']}")
            print(f"{'='*50}")
            
            # Train each model
            for model_name, model in self.models.items():
                optimizer = self.optimizers[model_name]
                scheduler = self.schedulers[model_name]
                
                train_loss, val_loss = self.train_single_model(model_name, model, optimizer, scheduler)
                
                training_history[model_name]['train_loss'].append(train_loss)
                training_history[model_name]['val_loss'].append(val_loss)
            
            # Update ensemble weights
            if epoch % 5 == 0:
                self.update_ensemble_weights()
        
        print("\nTraining completed!")
        return training_history
    
    def predict_single_model(self, model):
        """Generate predictions from a single model"""
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for X, _, org_indices in self.predict_dataloader:
                X = X.to(self.device)
                predictions = model(X)
                pred_np = predictions.cpu().numpy()
                
                # Unnormalize predictions
                pred_unnormalized = self.predict_dataloader.dataset.getOriginalSpacePredictions(
                    pred_np, org_indices
                )
                all_predictions.append(pred_unnormalized)
        
        return np.concatenate(all_predictions, axis=0)
    
    def predict_ensemble(self):
        """Generate ensemble predictions"""
        print("Generating ensemble predictions...")
        
        # Get predictions from each model
        model_predictions = {}
        for model_name, model in self.models.items():
            print(f"  Generating predictions from {model_name}...")
            model_predictions[model_name] = self.predict_single_model(model)
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(model_predictions['transformer'])
        
        for model_name, predictions in model_predictions.items():
            weight = self.ensemble_weights[model_name]
            ensemble_pred += weight * predictions
            print(f"  {model_name}: weight = {weight:.4f}")
        
        # Save individual model predictions
        for model_name, predictions in model_predictions.items():
            self.save_predictions(predictions, f"ensemble_{model_name}_submission.csv")
        
        # Save ensemble predictions
        self.save_predictions(ensemble_pred, "architecture_ensemble_submission.csv")
        
        print(f"Ensemble predictions saved! Shape: {ensemble_pred.shape}")
        return ensemble_pred
    
    def save_predictions(self, predictions, filename):
        """Save predictions to CSV file"""
        assert predictions.shape == (2100, 60, 2), f"Expected shape (2100, 60, 2), got {predictions.shape}"
        
        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])
        output_df.index.name = "index"
        
        filepath = os.path.join(utils.SUBMISSION_DIR, filename)
        output_df.to_csv(filepath)
        print(f"Saved predictions to {filepath}")

def main():
    """Main function for architecture ensemble"""
    config = {
        "BATCH_SIZE": 16,  # Reduced for multiple models
        "LEARNING_RATE": 0.001,
        "EPOCHS": 50,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.1,
        "NUM_SAMPLES": 10000,
        
        # Transformer parameters
        "NUM_QUERIES": 2,
        #"D_INPUT": 50 * 110 * 6,
        "D_INPUT": 50 * 50 * 9,
        "D_MODEL": 256,  # Reduced for efficiency
        "N_HEAD": 4,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.2,
        "D_EGO_FFNN_INPUT": 4,
        "FFNN_D_HIDDEN": 512,
        "FFNN_NUM_HIDDEN_LAYERS": 3,
        "D_OUTPUT": 60 * 2,
    }
    
    print("Initializing Architecture Ensemble...")
    trainer = ArchitectureEnsembleTrainer(config)
    
    print("Starting ensemble training...")
    training_history = trainer.train_ensemble()
    
    print("Generating final predictions...")
    ensemble_predictions = trainer.predict_ensemble()
    
    print("Architecture ensemble training and prediction completed!")
    
    #return trainer, training_history, ensemble_predictions
