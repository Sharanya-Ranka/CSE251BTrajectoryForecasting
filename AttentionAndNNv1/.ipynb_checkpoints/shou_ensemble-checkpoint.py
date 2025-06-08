import AttentionAndNN.data as dataset
from torch.utils.data import DataLoader
import AttentionAndNN.model as model
import AttentionAndNN.data_create as data_create
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import utilities as utils

def ensure_preprocessed_data_exists():
        data_dir = os.path.join("Data", "IntermediateData", "AttentionAndNN")
        train_file = os.path.join(data_dir, "train.npz")
        test_file = os.path.join(data_dir, "test.npz")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print("[INFO] Preprocessing data using data_create.main() ...")
            data_create.main()
        else:
            print("[INFO] Found preprocessed data. Skipping preprocessing.")

class SimpleNNTrainer:
    def __init__(self, config, model_suffix=""):
        self.config = config
        self.best_eval_loss = float('inf')
        self.model_suffix = model_suffix  # To distinguish between models

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

        # Apply acceleration-based filtering if specified
        # if hasattr(self.config, 'ACCELERATION_INDICES') and self.config['ACCELERATION_INDICES'] is not None:
        #     train_indices = self.config['ACCELERATION_INDICES']

        train_config = {
            "INDICES": train_indices,
            "DATA_FILENAME": "train.npz",
        }
        test_config = {"INDICES": test_indices, "DATA_FILENAME": "train.npz"}

        self.train_dataset = dataset.AllAgentsNormalizedDataset(train_config)
        self.test_dataset = dataset.AllAgentsNormalizedDataset(test_config)

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

        self.predict_dataset = dataset.AllAgentsNormalizedDataset(predict_config)

        self.predict_dataloader = DataLoader(
            self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

    def setUpModel(self):
        self.model = model.AttentionAndNN(self.config)
        self.model.to(self.config["DEVICE"])
        print(f"Model {self.model_suffix}:")
        print(self.model)

    def setUpOptimizer(self):
        # Define the optimizer being used
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["LEARNING_RATE"]
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.2
        )

    def train(self):
        training_progress = {}
        best_unnormalized_score = 20
        for epoch in range(self.config["EPOCHS"]):
            train_loss = self.train_epoch()
            eval_loss = self.eval_epoch()
            print(
                f"Model {self.model_suffix} - Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
            )
            training_progress[epoch] = {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }

            self.scheduler.step(eval_loss)
            print(
                f"Model {self.model_suffix} - Epoch {epoch+1}, Current Learning Rate: {self.scheduler.get_last_lr()[0]}"
            )

            if self.config.get("ANALYZE"):
                origspace_score_tr = self.analyze(self.train_dataloader)
                print(f"Model {self.model_suffix} - OrigSpace score (train) {origspace_score_tr}")

                origspace_score_te = self.analyze(self.test_dataloader)
                print(f"Model {self.model_suffix} - OrigSpace score (test) {origspace_score_te}")

                if origspace_score_te < best_unnormalized_score:
                    best_unnormalized_score = origspace_score_te
                    self.predict()

        return training_progress
    
    def weighted_mse_loss(self, pred, target, weights):
        diff = (pred-target) **2
        weighted_diff = diff*weights
        return weighted_diff.mean()

    def computeLoss(self, true, prediction):
        #weights = torch.tensor([2.0, 2.0, 0.75, 0.75, 1.0], device=true.device)
        weights = torch.tensor([2.0, 2.0], device=true.device)
        loss = self.weighted_mse_loss(prediction, true, weights)
        return loss

    # Training function
    def train_epoch(self):
        num_batches = len(self.train_dataloader)

        self.model.train()

        train_loss, total = 0, 0
        for batch, (X, y, _) in enumerate(self.train_dataloader):
            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

            # Compute prediction error
            prediction = self.model(X)

            puf = prediction.reshape(y.shape)
            loss = self.computeLoss(y, puf)
            train_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        average_train_loss = float(train_loss / num_batches)
        return average_train_loss

    def eval_epoch(self):
        # Set the model to evaluation mode.
        self.model.eval()

        eval_loss = 0
        num_batches = len(self.test_dataloader)

        # Disable gradient computation.
        with torch.no_grad():
            for X, y, _ in self.test_dataloader:
                # Move data to the specified device
                X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

                # Compute prediction error
                prediction = self.model(X)

                puf = prediction.reshape(y.shape)
                loss = self.computeLoss(y, puf)
                eval_loss += loss.item()

        # Calculate average loss
        average_eval_loss = float(eval_loss / num_batches)

        return average_eval_loss

    def analyze(self, dataloader: DataLoader):
        #num_batches = self.config["ANALYZE_NUM_EXAMPLES"]  / self.config["BATCH_SIZE"]
        num_batches = self.config["ANALYZE_NUM_BATCHES"]

        self.model.eval()
        inference_steps = 60

        total_unnormalized = 0
        for batch, (X, y, org_indices) in enumerate(dataloader):
            if batch > num_batches:
                break

            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

            # Compute prediction error
            prediction = self.model(X)

            puf = prediction.reshape(y.shape)

            true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                y.detach().cpu().numpy(), org_indices
            )
            pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                puf.detach().cpu().numpy(), org_indices
            )
            unnormalized_metric = dataloader.dataset.computeOriginalSpaceMetric(
                true_unnormalized, pred_unnormalized
            )
            total_unnormalized += unnormalized_metric

        return total_unnormalized / min(num_batches, len(dataloader))

    def predict(self):
        all_predictions = []
        dataloader = self.predict_dataloader
        inference_steps = 60

        self.model.eval()
        with torch.no_grad():
            for batch, (X, _, org_indices) in enumerate(dataloader):
                X = X.to(self.config["DEVICE"])
                
                # Get initial prediction
                predictions = self.model(X)
                #predictions = predictions.reshape(-1, 60, 5)
                
                # Convert to numpy for unnormalization
                pred_np = predictions.cpu().numpy()
                
                # Unnormalize predictions
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(pred_np, org_indices)
                all_predictions.append(pred_unnormalized)

        all_np_predictions = np.concatenate(all_predictions, axis=0)[:, :, :2]
        self.convertAndSavePredictions(all_np_predictions)

    def convertAndSavePredictions(self, predictions):
        assert tuple(predictions.shape) == (2100, 60, 2)

        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])

        output_df.index.name = "index"
        
        # Create Submissions directory if it doesn't exist
        os.makedirs(utils.SUBMISSION_DIR, exist_ok=True)
        
        # Save individual model predictions
        filename = f"simple_nn_submission{self.model_suffix}.csv"
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, filename))
        print(f"Saved predictions for model {self.model_suffix} to {filename}")

    

# Call this before loading npz files
ensure_preprocessed_data_exists()
class AccelerationBasedEnsembleTrainer:
    def __init__(self, config):
        self.config = config
        
    def calculate_mean_acceleration(self, data_indices):
        """Calculate mean acceleration for given data indices"""
        # Load the training data
        data_path = os.path.join(utils.INTERMEDIATE_DATA_DIR, "NewTransformer", "train.npz")
        data_file = np.load(data_path)
        
        # Get the data for specified indices
        data_subset = data_file["data"][data_indices]
        
        accelerations = []
        for sample in data_subset:
            # Assuming the data structure contains velocity information
            # Calculate acceleration from velocity differences
            # You may need to adjust this based on your actual data structure
            
            # For this example, assuming data contains position/velocity info
            # and we calculate acceleration magnitude
            velocities = sample[:, 2:4] if sample.shape[1] >= 4 else sample[:, :2]  # x_vel, y_vel
            
            # Calculate acceleration as velocity differences
            if len(velocities) > 1:
                vel_diffs = np.diff(velocities, axis=0)
                accel_magnitudes = np.sqrt(np.sum(vel_diffs**2, axis=1))
                mean_accel = np.mean(accel_magnitudes)
            else:
                mean_accel = 0.0
                
            accelerations.append(mean_accel)
        
        return np.array(accelerations)
    
    def split_data_by_acceleration(self):
        """Split training data into two halves based on mean acceleration"""
        total_indices = np.arange(self.config["NUM_SAMPLES"])
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )
        
        # Calculate mean acceleration for training data
        mean_accelerations = self.calculate_mean_acceleration(train_indices)
        
        # Find median acceleration to split data
        median_accel = np.median(mean_accelerations)
        
        # Split indices based on acceleration
        low_accel_mask = mean_accelerations <= median_accel
        high_accel_mask = mean_accelerations > median_accel
        
        low_accel_indices = train_indices[low_accel_mask]
        high_accel_indices = train_indices[high_accel_mask]
        
        print(f"Data split summary:")
        print(f"Total training samples: {len(train_indices)}")
        print(f"Low acceleration samples: {len(low_accel_indices)} (mean accel <= {median_accel:.4f})")
        print(f"High acceleration samples: {len(high_accel_indices)} (mean accel > {median_accel:.4f})")
        
        return low_accel_indices, high_accel_indices, test_indices
    
    def train_ensemble(self):
        """Train two models on different acceleration subsets"""
        # Split data
        low_accel_indices, high_accel_indices, test_indices = self.split_data_by_acceleration()
        
        # Create configs for both models
        low_accel_config = self.config.copy()
        high_accel_config = self.config.copy()
        
        low_accel_config['ACCELERATION_INDICES'] = low_accel_indices
        high_accel_config['ACCELERATION_INDICES'] = high_accel_indices
        
        # Train model 1 (low acceleration)
        # print("\n" + "="*50)
        # print("Training Model 1 (Low Acceleration)")
        # print("="*50)
        # trainer1 = SimpleNNTrainer(low_accel_config, model_suffix="_low_accel")
        # progress1 = trainer1.performPipeline()
        
        # Train model 2 (high acceleration)
        print("\n" + "="*50)
        print("Training Model 2 (High Acceleration)")
        print("="*50)
        trainer2 = SimpleNNTrainer(high_accel_config, model_suffix="_high_accel")
        progress2 = trainer2.performPipeline()
        
        # Create ensemble predictions
        self.create_ensemble_predictions()
        
        #return progress1, progress2
        return progress2
    
    def create_ensemble_predictions(self):
        """Average predictions from both models and create final submission"""
        try:
            # Load predictions from both models
            low_accel_file = os.path.join(utils.SUBMISSION_DIR, "simple_nn_submission_low_accel.csv")
            high_accel_file = os.path.join(utils.SUBMISSION_DIR, "simple_nn_submission_high_accel.csv")
            
            df_low = pd.read_csv(low_accel_file, index_col=0)
            df_high = pd.read_csv(high_accel_file, index_col=0)
            
            print(f"Loaded predictions from {low_accel_file}")
            print(f"Loaded predictions from {high_accel_file}")
            
            # Calculate mean of predictions
            ensemble_predictions = (df_low + df_high) / 2
            
            # Save ensemble predictions
            ensemble_file = os.path.join(utils.SUBMISSION_DIR, "ensemble_submission.csv")
            ensemble_predictions.to_csv(ensemble_file)
            
            print(f"\nEnsemble predictions saved to: {ensemble_file}")
            print(f"Ensemble shape: {ensemble_predictions.shape}")
            
            # Optional: Save some statistics
            stats_file = os.path.join(utils.SUBMISSION_DIR, "ensemble_stats.txt")
            with open(stats_file, 'w') as f:
                f.write("Ensemble Statistics\n")
                f.write("==================\n")
                f.write(f"Low acceleration model predictions shape: {df_low.shape}\n")
                f.write(f"High acceleration model predictions shape: {df_high.shape}\n")
                f.write(f"Ensemble predictions shape: {ensemble_predictions.shape}\n")
                f.write(f"Mean x prediction: {ensemble_predictions['x'].mean():.6f}\n")
                f.write(f"Mean y prediction: {ensemble_predictions['y'].mean():.6f}\n")
                f.write(f"Std x prediction: {ensemble_predictions['x'].std():.6f}\n")
                f.write(f"Std y prediction: {ensemble_predictions['y'].std():.6f}\n")
            
            print(f"Ensemble statistics saved to: {stats_file}")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find prediction files. Make sure both models completed training and prediction.")
            print(f"Missing file: {e}")
        except Exception as e:
            print(f"Error creating ensemble: {e}")


def main():
    config = {
        # "BATCH_SIZE": 64,
        # "LEARNING_RATE": 0.001,
        # "EPOCHS": 100,
        # "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        # "TEST_SIZE": 0.2,
        # "NUM_SAMPLES": 10000,
        # # Transformer specific parameters
        # "D_INPUT": 1 * 50 * 5,
        # "D_OUTPUT": 1 * 60 * 5,
        # "D_HIDDEN": 5 * 50 * 5,
        # "NUM_HIDDEN_LAYERS": 6,
        # "NUM_QUERIES": 2,
        # # Analysis parameters (optional based on ANALYZE flag)
        # "ANALYZE": True,
        # "ANALYZE_NUM_EXAMPLES": 100,
        # "D_MODEL": 128,              # Model dimension
        # "NHEAD": 8,                  # Number of attention heads
        # "NUM_ENCODER_LAYERS": 6,     # Number of encoder layers
        # "NUM_DECODER_LAYERS": 6,     # Number of decoder layers  
        # "DIM_FEEDFORWARD": 512,      # Feedforward dimension
        # "DROPOUT": 0.1,              # Dropout rate
         "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.005,
        "EPOCHS": 100,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.1,
        "NUM_SAMPLES": 10000,
        # Transformer specific parameters
        "NUM_QUERIES" : 2,
        # "D_INPUT": 50 * 9 + 50 * 6 + 7,
        "D_INPUT": 50 * 9,
        "D_MODEL": 500,
        "N_HEAD": 2,
        "NUM_LAYERS": 5,
        "DROPOUT": 0.2,  # Example dropout rate
        # FFNN Specific parameters
        "D_EGO_FFNN_INPUT" : 4,
        "FFNN_D_HIDDEN" : 1000,
        "FFNN_NUM_HIDDEN_LAYERS":5,
        "D_OUTPUT": 60 * 2,
        
        # Analysis parameters (optional based on ANALYZE flag)
        "ANALYZE": True,
        "ANALYZE_NUM_BATCHES": 50,
    }

    # Use ensemble trainer instead of single trainer
    ensure_preprocessed_data_exists()
    ensemble_trainer = AccelerationBasedEnsembleTrainer(config)
    #progress1, 
    progress2 = ensemble_trainer.train_ensemble()
    
    # Plot training progress for both models
    #df_progress1 = pd.DataFrame.from_dict(progress1, orient="index")
    df_progress2 = pd.DataFrame.from_dict(progress2, orient="index")

    # Plotting
    # plt.figure(figsize=(15, 6))
    
    # # Plot for low acceleration model
    # plt.subplot(1, 2, 1)
    # df_progress1[["train_loss", "eval_loss"]].plot(ax=plt.gca())
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Low Acceleration Model - Training and Validation Loss")
    # plt.grid(True)
    
    # Plot for high acceleration model
    plt.subplot(1, 2, 2)
    df_progress2[["train_loss", "eval_loss"]].plot(ax=plt.gca())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("High Acceleration Model - Training and Validation Loss")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("ensemble_loss_curves.png")
    plt.show()
    
    print("\nTraining completed!")
    print("Check the Submissions folder for:")
    print("- simple_nn_submission_low_accel.csv (Low acceleration model predictions)")
    print("- simple_nn_submission_high_accel.csv (High acceleration model predictions)")
    print("- ensemble_submission.csv (Final ensemble predictions)")
    print("- ensemble_stats.txt (Ensemble statistics)")

