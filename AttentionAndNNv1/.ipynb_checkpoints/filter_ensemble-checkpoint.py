# Enhanced version of your training script with Kalman filter integration

import AttentionAndNN.data as dataset
from torch.utils.data import DataLoader
import AttentionAndNN.model as model
import AttentionAndNN.data_create as data_create
from AttentionAndNN.shou_ensemble import SimpleNNTrainer 
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import utilities as utils
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class TrajectoryKalmanFilter:
    """Kalman Filter for trajectory smoothing"""
    
    def __init__(self, dt=0.1, process_noise=1.0, measurement_noise=0.1):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        # Process noise covariance
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise, block_size=2)
        
        # Measurement noise covariance
        self.kf.R = np.eye(2) * measurement_noise
        
        # Initial state covariance
        self.kf.P *= 100
        
    def initialize_state(self, initial_positions):
        """Initialize Kalman filter state"""
        if len(initial_positions) >= 2:
            vel = (initial_positions[-1] - initial_positions[-2]) / self.dt
            self.kf.x = np.array([initial_positions[-1][0], initial_positions[-1][1], 
                                 vel[0], vel[1]])
        else:
            self.kf.x = np.array([initial_positions[-1][0], initial_positions[-1][1], 0, 0])
    
    def predict_and_update(self, measurement):
        """Predict and update with measurement"""
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x[:2].copy()

class KalmanPostProcessor:
    """Post-process predictions with Kalman filtering"""
    
    def __init__(self, dt=0.1, process_noise=0.5, measurement_noise=1.0):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def smooth_predictions(self, raw_predictions, input_trajectory):
        """Apply Kalman smoothing to raw predictions"""
        batch_size = raw_predictions.shape[0]
        smoothed_predictions = np.zeros_like(raw_predictions)
        
        print(f"Debug - raw_predictions shape: {raw_predictions.shape}")
        print(f"Debug - input_trajectory shape: {input_trajectory.shape}")
        
        for i in range(batch_size):
            kf = TrajectoryKalmanFilter(
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )
            
            # Get trajectory for this batch item
            input_traj = input_trajectory[i]  # Shape should be (timesteps, 2)
            print(f"Debug - Single trajectory shape: {input_traj.shape}")
            
            # Ensure input_traj is 2D with shape (timesteps, 2)
            if len(input_traj.shape) == 1:
                # If flattened, reshape assuming it contains x,y pairs
                if input_traj.shape[0] % 2 == 0:
                    input_traj = input_traj.reshape(-1, 2)
                else:
                    print(f"Warning: Cannot reshape trajectory of length {input_traj.shape[0]} into x,y pairs")
                    smoothed_predictions[i] = raw_predictions[i]
                    continue
            elif input_traj.shape[1] != 2:
                # If more than 2 features, take only the first 2 (assuming x, y)
                input_traj = input_traj[:, :2]
            
            # Find valid (non-zero) points for initialization
            non_zero_mask = np.any(input_traj != 0, axis=1)
            
            if np.any(non_zero_mask):
                valid_points = input_traj[non_zero_mask]
                # Use last few valid points for initialization (up to 5)
                init_points = valid_points[-min(5, len(valid_points)):]
                kf.initialize_state(init_points)
                
                # Apply Kalman filtering to each prediction step
                for t in range(raw_predictions.shape[1]):
                    smoothed_pos = kf.predict_and_update(raw_predictions[i, t])
                    smoothed_predictions[i, t] = smoothed_pos
            else:
                # Fallback to raw predictions if no valid initialization points
                print(f"Warning: No valid points found for trajectory {i}, using raw predictions")
                smoothed_predictions[i] = raw_predictions[i]
        
        return smoothed_predictions

class KalmanEnhancedSimpleNNTrainer(SimpleNNTrainer):
    """Enhanced trainer with Kalman filter post-processing"""
    
    def __init__(self, config, model_suffix="", use_kalman=True, kalman_params=None):
        super().__init__(config)
        self.use_kalman = use_kalman
        
        # Default Kalman parameters
        default_kalman_params = {
            'dt': 0.1,
            'process_noise': 0.5,
            'measurement_noise': 1.0
        }
        
        if kalman_params:
            default_kalman_params.update(kalman_params)
        
        self.kalman_processor = KalmanPostProcessor(**default_kalman_params)
        print(f"Kalman filter {'enabled' if use_kalman else 'disabled'} for model {model_suffix}")
    
    def extract_input_trajectory(self, X_batch):
        """Extract trajectory data from input batch for Kalman initialization"""
        batch_size = X_batch.shape[0]
        
        print(f"Debug - X_batch shape: {X_batch.shape}")
        
        # Convert to numpy for easier manipulation
        X_np = X_batch.cpu().numpy()
        
        # Assuming input is structured as (batch, features) where features represent
        # flattened trajectory data: 50 timesteps * 9 features
        # We need to extract x,y positions from each timestep
        
        if len(X_np.shape) == 2:
            # Input is flattened: (batch, 50*9)
            expected_features = 50 * 9  # 50 timesteps, 9 features each
            
            if X_np.shape[1] == expected_features:
                # Reshape to (batch, 50, 9)
                X_reshaped = X_np.reshape(batch_size, 50, 9)
                # Extract x, y positions (assuming they're the first 2 features)
                trajectory = X_reshaped[:, :, :2]
            else:
                print(f"Warning: Unexpected feature count {X_np.shape[1]}, expected {expected_features}")
                # Try to infer the structure
                if X_np.shape[1] % 50 == 0:
                    features_per_timestep = X_np.shape[1] // 50
                    X_reshaped = X_np.reshape(batch_size, 50, features_per_timestep)
                    # Take first 2 features as x, y
                    trajectory = X_reshaped[:, :, :2]
                else:
                    print("Cannot determine trajectory structure, using zeros")
                    trajectory = np.zeros((batch_size, 50, 2))
        
        elif len(X_np.shape) == 3:
            # Already properly shaped: (batch, timesteps, features)
            trajectory = X_np[:, :, :2]  # Extract x, y positions
        
        else:
            print(f"Unexpected input shape: {X_np.shape}")
            trajectory = np.zeros((batch_size, 50, 2))
        
        print(f"Debug - Extracted trajectory shape: {trajectory.shape}")
        return trajectory
    
    def predict(self):
        """Enhanced prediction with Kalman filtering"""
        all_predictions = []
        all_input_trajectories = []
        dataloader = self.predict_dataloader
        
        self.model.eval()
        with torch.no_grad():
            for batch, (X, _, org_indices) in enumerate(dataloader):
                X = X.to(self.config["DEVICE"])
                
                # Get raw NN predictions
                predictions = self.model(X)
                pred_np = predictions.cpu().numpy()
                
                # Extract input trajectory for Kalman initialization
                input_traj = self.extract_input_trajectory(X)
                
                # Unnormalize predictions
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(pred_np, org_indices)
                
                all_predictions.append(pred_unnormalized)
                all_input_trajectories.append(input_traj)
        
        # Concatenate all predictions
        all_np_predictions = np.concatenate(all_predictions, axis=0)
        all_input_trajectories = np.concatenate(all_input_trajectories, axis=0)
        
        print(f"Debug - Final predictions shape: {all_np_predictions.shape}")
        print(f"Debug - Final trajectories shape: {all_input_trajectories.shape}")
        
        if self.use_kalman:
            print("Applying Kalman filter smoothing...")
            
            # Ensure predictions are in the right shape for Kalman processing
            if len(all_np_predictions.shape) == 2:
                # If flattened, reshape to (batch, timesteps, 2)
                if all_np_predictions.shape[1] == 120:  # 60 timesteps * 2 coordinates
                    reshaped_preds = all_np_predictions.reshape(-1, 60, 2)
                else:
                    print(f"Warning: Cannot reshape predictions of shape {all_np_predictions.shape}")
                    reshaped_preds = all_np_predictions
            else:
                reshaped_preds = all_np_predictions
            
            # Apply Kalman smoothing
            if len(reshaped_preds.shape) == 3 and reshaped_preds.shape[2] >= 2:
                smoothed_predictions = self.kalman_processor.smooth_predictions(
                    reshaped_preds[:, :, :2],  # Ensure only x, y coordinates
                    all_input_trajectories
                )
            else:
                print("Warning: Cannot apply Kalman smoothing due to shape mismatch")
                smoothed_predictions = reshaped_preds
            
            final_predictions = smoothed_predictions
            print(f"Kalman smoothing applied. Shape: {final_predictions.shape}")
        else:
            # Ensure proper shape for non-Kalman predictions
            if len(all_np_predictions.shape) == 2 and all_np_predictions.shape[1] == 120:
                final_predictions = all_np_predictions.reshape(-1, 60, 2)
            elif len(all_np_predictions.shape) == 3:
                final_predictions = all_np_predictions[:, :, :2]
            else:
                final_predictions = all_np_predictions
        
        self.convertAndSavePredictions(final_predictions)

class KalmanEnsembleTrainer:
    """Enhanced ensemble trainer with Kalman filtering"""
    
    def __init__(self, config, kalman_params=None):
        self.config = config
        self.kalman_params = kalman_params or {
            'dt': 0.1,
            'process_noise': 0.5,
            'measurement_noise': 1.0
        }
    
    def calculate_mean_acceleration(self, data_indices):
        """Calculate mean acceleration for given data indices"""
        data_path = os.path.join(utils.INTERMEDIATE_DATA_DIR, "NewTransformer", "train.npz")
        data_file = np.load(data_path)
        data_subset = data_file["data"][data_indices]
        
        accelerations = []
        for sample in data_subset:
            velocities = sample[:, 2:4] if sample.shape[1] >= 4 else sample[:, :2]
            
            if len(velocities) > 1:
                vel_diffs = np.diff(velocities, axis=0)
                accel_magnitudes = np.sqrt(np.sum(vel_diffs**2, axis=1))
                mean_accel = np.mean(accel_magnitudes)
            else:
                mean_accel = 0.0
                
            accelerations.append(mean_accel)
        
        return np.array(accelerations)
    
    def split_data_by_acceleration(self):
        """Split training data by acceleration"""
        total_indices = np.arange(self.config["NUM_SAMPLES"])
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )
        
        mean_accelerations = self.calculate_mean_acceleration(train_indices)
        median_accel = np.median(mean_accelerations)
        
        low_accel_mask = mean_accelerations <= median_accel
        high_accel_mask = mean_accelerations > median_accel
        
        low_accel_indices = train_indices[low_accel_mask]
        high_accel_indices = train_indices[high_accel_mask]
        
        print(f"Data split summary:")
        print(f"Total training samples: {len(train_indices)}")
        print(f"Low acceleration samples: {len(low_accel_indices)} (mean accel <= {median_accel:.4f})")
        print(f"High acceleration samples: {len(high_accel_indices)} (mean accel > {median_accel:.4f})")
        
        return low_accel_indices, high_accel_indices, test_indices
    
    def train_ensemble_with_kalman(self, use_kalman=True):
        """Train ensemble with Kalman filtering option"""
        low_accel_indices, high_accel_indices, test_indices = self.split_data_by_acceleration()
        
        # Create configs for both models
        low_accel_config = self.config.copy()
        high_accel_config = self.config.copy()
        
        low_accel_config['ACCELERATION_INDICES'] = low_accel_indices
        high_accel_config['ACCELERATION_INDICES'] = high_accel_indices
        
        # Train model 1 (low acceleration) with Kalman
        print("\n" + "="*50)
        print("Training Model 1 (Low Acceleration) with Kalman Filter")
        print("="*50)
        trainer1 = KalmanEnhancedSimpleNNTrainer(
            low_accel_config, 
            model_suffix="_low_accel_kalman" if use_kalman else "_low_accel",
            use_kalman=use_kalman,
            kalman_params=self.kalman_params
        )
        progress1 = trainer1.performPipeline()
        
        # Train model 2 (high acceleration) with Kalman
        print("\n" + "="*50)
        print("Training Model 2 (High Acceleration) with Kalman Filter")
        print("="*50)
        trainer2 = KalmanEnhancedSimpleNNTrainer(
            high_accel_config, 
            model_suffix="_high_accel_kalman" if use_kalman else "_high_accel",
            use_kalman=use_kalman,
            kalman_params=self.kalman_params
        )
        progress2 = trainer2.performPipeline()
        
        # Create ensemble predictions
        self.create_ensemble_predictions(use_kalman)
        
        return progress1, progress2
    
    def create_ensemble_predictions(self, use_kalman=True):
        """Create ensemble predictions"""
        suffix = "_kalman" if use_kalman else ""
        
        try:
            low_accel_file = os.path.join(utils.SUBMISSION_DIR, f"simple_nn_submission_low_accel{suffix}.csv")
            high_accel_file = os.path.join(utils.SUBMISSION_DIR, f"simple_nn_submission_high_accel{suffix}.csv")
            
            df_low = pd.read_csv(low_accel_file, index_col=0)
            df_high = pd.read_csv(high_accel_file, index_col=0)
            
            print(f"Loaded predictions from {low_accel_file}")
            print(f"Loaded predictions from {high_accel_file}")
            
            # Calculate ensemble
            ensemble_predictions = (df_low + df_high) / 2
            
            # Save ensemble predictions
            ensemble_file = os.path.join(utils.SUBMISSION_DIR, f"ensemble_submission{suffix}.csv")
            ensemble_predictions.to_csv(ensemble_file)
            
            print(f"\nEnsemble predictions saved to: {ensemble_file}")
            print(f"Ensemble shape: {ensemble_predictions.shape}")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find prediction files: {e}")
        except Exception as e:
            print(f"Error creating ensemble: {e}")

def main():
    """Main function with Kalman filter integration"""
    
    config = {
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.005,
        "EPOCHS": 100,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.1,
        "NUM_SAMPLES": 10000,
        "NUM_QUERIES": 2,
        "D_INPUT": 50 * 9,
        "D_MODEL": 500,
        "N_HEAD": 2,
        "NUM_LAYERS": 5,
        "DROPOUT": 0.2,
        "D_EGO_FFNN_INPUT": 4,
        "FFNN_D_HIDDEN": 1000,
        "FFNN_NUM_HIDDEN_LAYERS": 5,
        "D_OUTPUT": 60 * 2,
        "ANALYZE": True,
        "ANALYZE_NUM_BATCHES": 50,
    }
    kalman_params = {
        "dt": 0.1,
        "process_noise": 0.5,
        "measurement_noise": 1.0
    }

    print("\nInitializing Kalman Ensemble Trainer...\n")
    ensemble_trainer = KalmanEnsembleTrainer(config, kalman_params)

    print("\nStarting training with Kalman filtering...\n")
    progress_low, progress_high = ensemble_trainer.train_ensemble_with_kalman(use_kalman=True)

    print("\nTraining complete.\n")
    
    # Kalman filter parameters