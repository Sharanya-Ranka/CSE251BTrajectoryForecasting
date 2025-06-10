import ConstantVelocityPlusNN.data as dataset
from torch.utils.data import DataLoader
#import ConstantVelocityPlusNN.model as model
import LSTMWeightBasedLoss.model2 as model
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd
import utilities as utils


# import ConstantVelocityPlusNN.data as dataset
# from torch.utils.data import DataLoader
# import torch
# import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# import utilities as utils
from LSTMWeightBasedLoss.model2 import EgoAgentTransformer, EgoAgentEnsembleModel

class EnhancedNNTrainer:
    def __init__(self, config):
        self.config = config
        self.best_eval_loss = float('inf')
        self.best_unnormalized_score = float('inf')

    def performPipeline(self):
        self.setUpData()
        self.setUpModel()
        self.setUpOptimizer()
        training_progress = self.train()
        return training_progress

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

        self.train_dataset = dataset.EgoAgentNormalizedDataset(train_config)
        self.test_dataset = dataset.EgoAgentNormalizedDataset(test_config)

        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config["BATCH_SIZE"], 
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.config["DEVICE"] == "cuda" else False
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=self.config["BATCH_SIZE"], 
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.config["DEVICE"] == "cuda" else False
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
            self.predict_dataset, 
            batch_size=self.config["BATCH_SIZE"], 
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.config["DEVICE"] == "cuda" else False
        )

    def setUpModel(self):
        if self.config.get("USE_ENSEMBLE", False):
            self.model = EgoAgentEnsembleModel(self.config)
        else:
            self.model = EgoAgentTransformer(self.config)
        
        self.model = self.model.to(self.config["DEVICE"])
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def setUpOptimizer(self):
        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config["LEARNING_RATE"],
            weight_decay=self.config.get("WEIGHT_DECAY", 1e-4),
            betas=(0.9, 0.999)
        )

        # Use cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.config.get("T_0", 10), 
            T_mult=self.config.get("T_MULT", 2),
            eta_min=self.config["LEARNING_RATE"] * 0.01
        )

    def weighted_mse_loss(self, pred, target, weights):
        """
        Weighted MSE loss with different weights for different output dimensions
        """
        diff = (pred - target) ** 2
        
        # Expand weights to match the shape
        if len(weights.shape) == 1:
            weights = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, 5)
        
        weighted_diff = diff * weights
        return weighted_diff.mean()

    def computeLoss(self, true, prediction):
        """
        Compute loss with emphasis on position and velocity accuracy
        """
        # Different weights for [pos_diff_x, pos_diff_y, vel_x, vel_y, heading]
        weights = torch.tensor([3.0, 3.0, 1.5, 1.5, 0.5], device=true.device)
        
        # Compute weighted MSE loss
        loss = self.weighted_mse_loss(prediction, true, weights)
        
        # Add L1 regularization for better generalization
        l1_lambda = self.config.get("L1_LAMBDA", 1e-6)
        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        loss += l1_lambda * l1_norm
        
        return loss

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        num_batches = len(self.train_dataloader)
        
        for batch_idx, (X, y, _) in enumerate(self.train_dataloader):
            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
            
            # Forward pass
            prediction = self.model(X)
            puf = prediction.reshape(y.shape)
            
            # Compute loss
            loss = self.computeLoss(y, puf)
            train_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        return train_loss / num_batches

    def eval_epoch(self):
        self.model.eval()
        eval_loss = 0
        num_batches = len(self.test_dataloader)
        
        with torch.no_grad():
            for X, y, _ in self.test_dataloader:
                X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
                
                prediction = self.model(X)
                puf = prediction.reshape(y.shape)
                loss = self.computeLoss(y, puf)
                eval_loss += loss.item()
        
        return eval_loss / num_batches

    def analyze(self, dataloader):
        """
        Analyze model performance in original space
        """
        self.model.eval()
        num_batches = min(
            int(self.config["ANALYZE_NUM_EXAMPLES"] / self.config["BATCH_SIZE"]),
            len(dataloader)
        )
        
        total_unnormalized = 0
        
        with torch.no_grad():
            for batch_idx, (X, y, org_indices) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
                
                prediction = self.model(X)
                puf = prediction.reshape(y.shape)
                
                # Convert to numpy for unnormalization
                y_np = y.cpu().numpy()
                puf_np = puf.cpu().numpy()
                
                true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    y_np, org_indices
                )
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    puf_np, org_indices
                )
                
                unnormalized_metric = dataloader.dataset.computeOriginalSpaceMetric(
                    true_unnormalized, pred_unnormalized
                )
                total_unnormalized += unnormalized_metric
        
        return total_unnormalized / num_batches

    def train(self):
        training_progress = {}
        
        for epoch in range(self.config["EPOCHS"]):
            print(f"\nEpoch {epoch + 1}/{self.config['EPOCHS']}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch()
            
            # Evaluation
            eval_loss = self.eval_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Eval Loss: {eval_loss:.6f}")
            print(f"Learning Rate: {current_lr:.8f}")
            
            # Store progress
            training_progress[epoch] = {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "learning_rate": current_lr
            }
            
            # Analysis in original space
            if self.config.get("ANALYZE", False):
                train_score = self.analyze(self.train_dataloader)
                test_score = self.analyze(self.test_dataloader)
                
                print(f"Original Space Score (Train): {train_score:.6f}")
                print(f"Original Space Score (Test): {test_score:.6f}")
                
                training_progress[epoch]["train_orig_score"] = train_score
                training_progress[epoch]["test_orig_score"] = test_score
                
                # Save best model
                if test_score < self.best_unnormalized_score:
                    self.best_unnormalized_score = test_score
                    print(f"New best model! Score: {test_score:.6f}")
                    self.save_model(f"best_model_epoch_{epoch}.pth")
                    self.predict()
            
            # Early stopping
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                patience_counter = 0
            else:
                patience_counter = getattr(self, 'patience_counter', 0) + 1
            
            self.patience_counter = patience_counter
            
            if patience_counter >= self.config.get("PATIENCE", 15):
                print(f"Early stopping at epoch {epoch}")
                break
        
        return training_progress

    def save_model(self, filename):
        """Save model checkpoint"""
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_score': self.best_unnormalized_score
        }, os.path.join("checkpoints", filename))

    def predict(self):
        """Generate predictions for test set"""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (X, _, org_indices) in enumerate(self.predict_dataloader):
                X = X.to(self.config["DEVICE"])
                
                # Get predictions
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

    def convertAndSavePredictions(self, predictions):
        """Convert predictions to submission format"""
        assert tuple(predictions.shape) == (2100, 60, 2)
        
        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])
        output_df.index.name = "index"
        
        # Create submission directory
        os.makedirs(utils.SUBMISSION_DIR, exist_ok=True)
        
        # Save predictions
        filename = f"transformer_acceleration_submission.csv"
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, filename))
        print(f"Predictions saved to {filename}")

def main():
    config = {
        # Data parameters
        "BATCH_SIZE": 32,  # Reduced for transformer
        "TEST_SIZE": 0.2,
        "NUM_SAMPLES": 10000,
        
        # Model parameters
        "D_MODEL": 256,
        "NHEAD": 8,
        "NUM_ENCODER_LAYERS": 6,
        "DROPOUT": 0.1,
        "USE_ENSEMBLE": False,  # Set to True for ensemble model
        
        # Training parameters
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "EPOCHS": 100,
        "PATIENCE": 15,
        "L1_LAMBDA": 1e-6,
        
        # Scheduler parameters
        "T_0": 10,
        "T_MULT": 2,
        
        # Device
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        
        # Analysis parameters
        "ANALYZE": True,
        "ANALYZE_NUM_EXAMPLES": 500,
    }
    
    print("Enhanced Transformer Training with Acceleration Features")
    print("=" * 60)
    print(f"Device: {config['DEVICE']}")
    print(f"Model: {'Ensemble' if config['USE_ENSEMBLE'] else 'Transformer'}")
    print(f"Batch Size: {config['BATCH_SIZE']}")
    print(f"Learning Rate: {config['LEARNING_RATE']}")
    print("=" * 60)
    
    trainer = EnhancedNNTrainer(config)
    training_progress = trainer.performPipeline()
    
    # Plot training progress
    df_progress = pd.DataFrame.from_dict(training_progress, orient="index")
    
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    df_progress[["train_loss", "eval_loss"]].plot(ax=plt.gca())
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(2, 2, 2)
    df_progress["learning_rate"].plot(ax=plt.gca())
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    
    # Original space scores
    if "train_orig_score" in df_progress.columns:
        plt.subplot(2, 2, 3)
        df_progress[["train_orig_score", "test_orig_score"]].plot(ax=plt.gca())
        plt.title("Original Space Scores")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("enhanced_training_progress.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTraining completed!")
    print(f"Best evaluation loss: {trainer.best_eval_loss:.6f}")
    if hasattr(trainer, 'best_unnormalized_score'):
        print(f"Best original space score: {trainer.best_unnormalized_score:.6f}")

if __name__ == "__main__":
    main()