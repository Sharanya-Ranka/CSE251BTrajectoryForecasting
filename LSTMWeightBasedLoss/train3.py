import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.nn.functional as F

from LSTMWeightBasedLoss.model3 import SimpleGNN, EnsembleModel

    
class GNNTrainer:
    """Enhanced trainer class for GNN and Transformer ensemble models"""
    
    def __init__(self, config):
        self.config = config
        self.best_eval_loss = float('inf')
        self.current_lr = config["LEARNING_RATE"]
        
    def performPipeline(self):
        """Main training pipeline"""
        self.setUpData()
        self.setUpModel()
        self.setUpOptimizer()
        training_progress = self.train()
        return training_progress
    
    def setUpData(self):
        """Set up training and validation data with enhanced data loading"""
        total_indices = np.arange(self.config["NUM_SAMPLES"])
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )

        # Enhanced data configuration with more options
        data_config = {
            "normalize": True,
            "add_relative_pos": True,
            "add_velocity": True,
            "add_acceleration": False
        }

        train_config = {
            "INDICES": train_indices,
            "DATA_FILENAME": "train.npz",
            **data_config
        }
        
        test_config = {
            "INDICES": test_indices, 
            "DATA_FILENAME": "train.npz",
            **data_config
        }

        try:
            import ConstantVelocityPlusNN.data as dataset
            # Initialize datasets with enhanced features
            self.train_dataset = dataset.EgoAgentNormalizedDataset(train_config)
            self.test_dataset = dataset.EgoAgentNormalizedDataset(test_config)
            
            # Add data augmentation to training set
            if self.config.get("DATA_AUGMENTATION", False):
                self.train_dataset = self._add_augmentations(self.train_dataset)
        except ImportError as e:
            print(f"Could not import dataset: {e}")
            raise
        
        # Configure DataLoaders with more options
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.config.get("NUM_WORKERS", 0),
            pin_memory=True,
            drop_last=True
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.config.get("NUM_WORKERS", 0),
            pin_memory=True
        )

        # Prediction dataset setup
        predict_config = {
            "INDICES": np.arange(2100),
            "DATA_FILENAME": "test.npz",
            "INFERENCE": True,
            **data_config
        }
        
        self.predict_dataset = dataset.EgoAgentNormalizedDataset(predict_config)
        self.predict_dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False,
            num_workers=0
        )
        
        #self._debug_data_shapes()
    def weighted_mse_loss(self, prediction, target, weights):
        """
        Compute weighted mean squared error loss.

        Args:
            prediction: Predicted tensor, shape (batch, time, features)
            target: Ground truth tensor, same shape as prediction
            weights: Tensor of shape (features,) representing per-feature weights

        Returns:
            Scalar weighted MSE loss
        """
        # Ensure weights is broadcastable
        while weights.dim() < prediction.dim():
            weights = weights.unsqueeze(0)
        
        loss = weights * (prediction - target) ** 2
        return loss.mean()

    def _add_augmentations(self, dataset):
        """Add data augmentations to training dataset"""
        class AugmentedDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset):
                self.base_dataset = base_dataset
                
            def __len__(self):
                return len(self.base_dataset)
                
            def __getitem__(self, idx):
                data = self.base_dataset[idx]
                x, y = data[0], data[1]
                
                # Random scaling
                if np.random.rand() < 0.3:
                    scale = np.random.uniform(0.9, 1.1)
                    x = x * scale
                    y = y * scale
                
                # Random noise
                if np.random.rand() < 0.3:
                    noise = torch.randn_like(x) * 0.01
                    x = x + noise
                
                return (x, y) + data[2:] if len(data) > 2 else (x, y)
        
        return AugmentedDataset(dataset)
    
    def setUpModel(self):
        """Set up model with enhanced initialization"""
        model_type = self.config.get("MODEL_TYPE", "ensemble").lower()
        
        if model_type == "ensemble":
            self.model = EnsembleModel(self.config)
            print("Using Ensemble of GNN and Transformer")
        elif model_type == "gnn":
            self.model = EgoAgentGNN(self.config) if self.config.get("USE_TORCH_GEOMETRIC", False) \
                       else SimpleGNN(self.config)
            print("Using GNN model only")
        elif model_type == "transformer":
            self.model = TrajectoryTransformer(self.config)
            print("Using Transformer model only")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize weights properly
        self._initialize_weights()
        
        self.model.to(self.config["DEVICE"])
        print(f"Model initialized on {self.config['DEVICE']}")
        
        # Enhanced model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Add gradient clipping norm
        self.grad_clip = self.config.get("GRAD_CLIP", 1.0)
    
    def _initialize_weights(self):
        """Initialize model weights properly"""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def setUpOptimizer(self):
        """Set up optimizer with enhanced options"""
        # Different learning rates for different components
        param_groups = []
        
        if isinstance(self.model, EnsembleModel):
            # GNN parameters
            param_groups.append({
                'params': self.model.gnn_model.parameters(),
                'lr': self.config["LEARNING_RATE"],
                'weight_decay': self.config.get("GNN_WEIGHT_DECAY", self.config["WEIGHT_DECAY"])
            })
            
            # Transformer parameters
            param_groups.append({
                'params': self.model.transformer_model.parameters(),
                'lr': self.config["LEARNING_RATE"],
                'weight_decay': self.config.get("TRANSFORMER_WEIGHT_DECAY", self.config["WEIGHT_DECAY"])
            })
            
            # Ensemble parameters
            param_groups.append({
                'params': [self.model.ensemble_weights, self.model.temperature],
                'lr': self.config.get("ENSEMBLE_LEARNING_RATE", self.config["LEARNING_RATE"] * 0.1),
                'weight_decay': 0.0  # No weight decay for weights
            })
        else:
            # Single model case
            param_groups.append({
                'params': self.model.parameters(),
                'lr': self.config["LEARNING_RATE"],
                'weight_decay': self.config["WEIGHT_DECAY"]
            })
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            eps=self.config.get("EPS", 1e-8),
            betas=self.config.get("BETAS", (0.9, 0.999))
        )
        
        # Enhanced learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config["SCHEDULER_FACTOR"],
            patience=self.config["SCHEDULER_PATIENCE"],
            verbose=True,
            min_lr=self.config.get("MIN_LR", 1e-6)
        )
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.get("MIXED_PRECISION", False))
    
    def computeLoss(self, true, prediction):
        """Enhanced loss computation with multiple terms"""
        # Shape handling
        if prediction.shape != true.shape:
            if prediction.numel() == true.numel():
                prediction = prediction.view(true.shape)
            else:
                raise ValueError(f"Shape mismatch: pred {prediction.shape}, true {true.shape}")
        
        # Base weighted MSE
        position_weight = self.config.get("POSITION_WEIGHT", 2.0)
        velocity_weight = self.config.get("VELOCITY_WEIGHT", 0.75)
        heading_weight = self.config.get("HEADING_WEIGHT", 1.0)
        
        weights = torch.tensor([
            position_weight, position_weight,  # x,y
            velocity_weight, velocity_weight,  # vx,vy
            heading_weight                     # heading
        ], device=true.device)[:true.shape[-1]]
        
        mse_loss = self.weighted_mse_loss(prediction, true, weights)

        
        # Additional loss terms
        total_loss = mse_loss
        
        # Velocity consistency loss
        if true.shape[-1] >= 4 and self.config.get("VELOCITY_LOSS_WEIGHT", 0.0) > 0:
            pred_vel = prediction[..., 2:4]
            true_vel = true[..., 2:4]
            vel_loss = F.mse_loss(pred_vel, true_vel)
            total_loss += self.config["VELOCITY_LOSS_WEIGHT"] * vel_loss
        
        # Heading consistency loss
        if true.shape[-1] >= 5 and self.config.get("HEADING_LOSS_WEIGHT", 0.0) > 0:
            pred_heading = prediction[..., 4]
            true_heading = true[..., 4]
            heading_loss = F.mse_loss(pred_heading, true_heading)
            total_loss += self.config["HEADING_LOSS_WEIGHT"] * heading_loss
            
        return total_loss
    
    def train_epoch(self):
        """Enhanced training loop with mixed precision and gradient clipping"""
        self.model.train()
        train_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, batch_data in enumerate(self.train_dataloader):
            X, y = batch_data[0], batch_data[1]
            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.config.get("MIXED_PRECISION", False)):
                prediction = self.model(X)
                loss = self.computeLoss(y, prediction)
            
            # Backward pass with scaling for mixed precision
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip,
                norm_type=2.0
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.get("LOG_INTERVAL", 50) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Batch {batch_idx}/{num_batches} | "
                    f"Loss: {loss.item():.6f} | "
                    f"LR: {current_lr:.2e}"
                )
                
                # Log ensemble weights if using ensemble
                if isinstance(self.model, EnsembleModel):
                    weights = F.softmax(self.model.ensemble_weights / self.model.temperature, dim=0)
                    print(f"  Ensemble weights: GNN {weights[0].item():.3f}, "
                          f"Transformer {weights[1].item():.3f}")
        
        return train_loss / num_batches
    
    def eval_epoch(self):
        """Enhanced evaluation with additional metrics"""
        self.model.eval()
        eval_loss = 0.0
        num_batches = len(self.test_dataloader)
        
        with torch.no_grad():
            for X, y, _ in self.test_dataloader:
                X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
                
                prediction = self.model(X)
                
                # Ensure proper shape
                if prediction.shape != y.shape:
                    prediction = prediction.view(y.shape)
                
                loss = self.computeLoss(y, prediction)
                eval_loss += loss.item()
        
        return eval_loss / num_batches
    
    def train(self):
        """Enhanced training loop with more comprehensive tracking"""
        training_progress = {
            'train_loss': [],
            'eval_loss': [],
            'lr': [],
            'best_epoch': 0
        }
        
        best_eval_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["EPOCHS"]):
            print(f"\nEpoch {epoch + 1}/{self.config['EPOCHS']}")
            print("-" * 50)
            
            # Train and evaluate
            train_loss = self.train_epoch()
            eval_loss = self.eval_epoch()
            
            # Update learning rate
            self.scheduler.step(eval_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.current_lr = current_lr
            
            # Store progress
            training_progress['train_loss'].append(train_loss)
            training_progress['eval_loss'].append(eval_loss)
            training_progress['lr'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.6f} | Eval Loss: {eval_loss:.6f} | LR: {current_lr:.2e}")
            
            # Checkpoint and early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                training_progress['best_epoch'] = epoch
                self.save_checkpoint(epoch, "best_model")
                patience_counter = 0
                
                # Generate predictions with best model
                if self.config.get("SAVE_PREDICTIONS", True):
                    self.predict()
            else:
                patience_counter += 1
                if patience_counter >= self.config["EARLY_STOPPING_PATIENCE"]:
                    print(f"Early stopping after {patience_counter} epochs without improvement")
                    break
            
            # Additional analysis
            if self.config.get("ANALYZE", False) and (epoch % 5 == 0 or epoch == self.config["EPOCHS"] - 1):
                self._run_analysis(epoch, training_progress)
        
        # Final evaluation
        print("\nTraining completed!")
        print(f"Best evaluation loss: {best_eval_loss:.6f} at epoch {training_progress['best_epoch'] + 1}")
        
        return training_progress
    
    def _run_analysis(self, epoch, training_progress):
        """Run periodic analysis during training"""
        try:
            # Original space metrics
            train_score = self.analyze(self.train_dataloader)
            test_score = self.analyze(self.test_dataloader)
            
            print(f"Original Space Metrics - Train: {train_score:.4f}, Test: {test_score:.4f}")
            
            # Store in progress
            if 'origspace_train' not in training_progress:
                training_progress['origspace_train'] = []
                training_progress['origspace_test'] = []
            
            training_progress['origspace_train'].append(train_score)
            training_progress['origspace_test'].append(test_score)
            
        except Exception as e:
            print(f"Analysis failed: {e}")
    
    # [Keep all other existing methods unchanged...]
    def analyze(self, dataloader):
        """Analyze model performance in original space"""
        self.model.eval()
        total_unnormalized = 0
        num_batches = min(
            self.config["ANALYZE_NUM_EXAMPLES"] // self.config["BATCH_SIZE"],
            len(dataloader)
        )
        
        with torch.no_grad():
            for batch_idx, (X, y, org_indices) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
                
                # Forward pass
                prediction = self.model(X)
                
                # Reshape if necessary
                if prediction.shape != y.shape:
                    prediction = prediction.reshape(y.shape)
                
                # Convert to original space
                true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    y.cpu().numpy(), org_indices
                )
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    prediction.cpu().numpy(), org_indices
                )
                
                # Compute metric
                unnormalized_metric = dataloader.dataset.computeOriginalSpaceMetric(
                    true_unnormalized, pred_unnormalized
                )
                total_unnormalized += unnormalized_metric
        
        return total_unnormalized / num_batches
    
    def predict(self):
        """Generate predictions for test set"""
        print("Generating predictions...")
        all_predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X, _, org_indices) in enumerate(self.predict_dataloader):
                X = X.to(self.config["DEVICE"])
                
                # Forward pass
                predictions = self.model(X)
                predictions = predictions.reshape(-1, 60, 5)
                
                # Convert to numpy and unnormalize
                pred_np = predictions.cpu().numpy()
                pred_unnormalized = self.predict_dataloader.dataset.getOriginalSpacePredictions(
                    pred_np, org_indices
                )
                all_predictions.append(pred_unnormalized)
        
        # Combine all predictions
        all_np_predictions = np.concatenate(all_predictions, axis=0)[:, :, :2]
        self.convertAndSavePredictions(all_np_predictions)
        print("Predictions saved!")
    
    def convertAndSavePredictions(self, predictions):
        """Convert and save predictions to CSV"""
        assert predictions.shape == (2100, 60, 2), f"Expected shape (2100, 60, 2), got {predictions.shape}"
        
        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])
        output_df.index.name = "index"
        
        # Create submissions directory
        os.makedirs("submissions", exist_ok=True)
        
        output_df.to_csv("submissions/gnn_submission.csv")
        print("Submission saved to submissions/gnn_submission.csv")
    
    def save_checkpoint(self, epoch, name):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, f"checkpoints/{name}.pth")
        print(f"Checkpoint saved: checkpoints/{name}.pth")

def main():
    """Main function to run GNN training"""
    # config = {
    #     "BATCH_SIZE": 16,  # Smaller batch size for GNN
    #     "LEARNING_RATE": 0.001,
    #     "EPOCHS": 150,
    #     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    #     "TEST_SIZE": 0.2,
    #     "NUM_SAMPLES": 10000,
        
    #     # GNN specific parameters
    #     "HIDDEN_DIM": 256,
    #     "NUM_MESSAGE_LAYERS": 3,
    #     "NUM_HEADS": 8,
    #     "DROPOUT": 0.1,
    #     "DISTANCE_THRESHOLD": 50.0,
    #     "USE_TORCH_GEOMETRIC": False,  # Set to True if you have torch_geometric installed
        
    #     # Training parameters
    #     "WEIGHT_DECAY": 1e-4,
    #     "SCHEDULER_PATIENCE": 10,
    #     "SCHEDULER_FACTOR": 0.5,
    #     "EARLY_STOPPING_PATIENCE": 25,
        
    #     # Analysis parameters
    #     "ANALYZE": True,
    #     "ANALYZE_NUM_EXAMPLES": 200,
    # }
    config = {
        "MODEL_TYPE": "ensemble",  # "ensemble", "gnn", or "transformer"
        "USE_TORCH_GEOMETRIC": False,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        # Data configuration
        "NUM_SAMPLES": 10000,
        "TEST_SIZE": 0.2,
        "DATA_AUGMENTATION": True,
        "HIDDEN_DIM": 256,
        # Training parameters
        "NUM_HEADS": 8,
        "BATCH_SIZE": 16,
        "EPOCHS": 10,
        "LEARNING_RATE": 0.0005,
        "WEIGHT_DECAY": 1e-5,
        "GRAD_CLIP": 1.0,
        "MIXED_PRECISION": True,
        "DROPOUT": 0.1,
        
        # Loss weights
        "POSITION_WEIGHT": 2.0,
        "VELOCITY_WEIGHT": 0.75,
        "HEADING_WEIGHT": 1.0,
        "VELOCITY_LOSS_WEIGHT": 0.1,
        "HEADING_LOSS_WEIGHT": 0.1,
        
        # Scheduler
        "SCHEDULER_PATIENCE": 15,
        "SCHEDULER_FACTOR": 0.5,
        "MIN_LR": 1e-6,
        
        # Early stopping
        "EARLY_STOPPING_PATIENCE": 30,
        
        # Analysis
        "ANALYZE": True,
        "ANALYZE_NUM_EXAMPLES": 200,
        "LOG_INTERVAL": 50
    }
    
    print("Starting GNN Training Pipeline")
    print(f"Device: {config['DEVICE']}")
    print(f"Batch Size: {config['BATCH_SIZE']}")
    print(f"Hidden Dimension: {config['HIDDEN_DIM']}")
    
    trainer = GNNTrainer(config)
    training_progress = trainer.performPipeline()
    
    # Plot training progress
    df_progress = pd.DataFrame.from_dict(training_progress, orient="index")
    
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    df_progress[["train_loss", "eval_loss"]].plot(ax=plt.gca())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    
    # Original space scores (if available)
    if "origspace_score_train" in df_progress.columns:
        plt.subplot(1, 3, 2)
        df_progress[["origspace_score_train", "origspace_score_test"]].plot(ax=plt.gca())
        plt.xlabel("Epoch")
        plt.ylabel("Original Space Score")
        plt.title("Original Space Performance")
        plt.grid(True)
        plt.legend()
    
    # Learning rate (if scheduler is used)
    plt.subplot(1, 3, 3)
    # This would need to be tracked during training if you want to plot it
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("gnn_training_progress.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training completed!")
    return training_progress

if __name__ == "__main__":
    # Run the training
    training_progress = main()