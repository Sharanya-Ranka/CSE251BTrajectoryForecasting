import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
import pandas as pd

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from current directory
from CSE251BTrajectoryForecasting import model
from CSE251BTrajectoryForecasting.data import AugmentedDataset
from CSE251BTrajectoryForecasting import utilities as utils
from CSE251BTrajectoryForecasting.gnn_config import MODEL_CONFIG

def main():
    # Model configuration
    config = {
        "BATCH_SIZE": 32,
        "DEVICE": "mps" if torch.backends.mps.is_available() else "cpu",
        **MODEL_CONFIG
    }

    # Initialize model
    print("Loading model...")
    model_instance = model.GNNTrajectoryModel(config)
    model_instance.to(config["DEVICE"])
    
    # Load saved weights
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model_instance.load_state_dict(torch.load(model_path, map_location=config["DEVICE"]))
    model_instance.eval()
    print("Model loaded successfully")

    # Set up prediction dataset
    print("Setting up prediction dataset...")
    predict_indices = np.arange(2100)
    predict_config = {
        "INDICES": predict_indices,
        "DATA_FILENAME": "test.npz",
        "INFERENCE": True,
        "WINDOW_SIZE": config["WINDOW_SIZE"],
        "STRIDE": config["STRIDE"]
    }

    predict_dataset = AugmentedDataset(predict_config)
    predict_dataloader = DataLoader(
        predict_dataset, 
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config["DEVICE"] == "cuda" else False
    )
    print(f"Prediction dataset size: {len(predict_dataset)}")

    # Generate predictions
    print("Generating predictions...")
    all_predictions = []
    num_ensembles = 2

    with torch.no_grad():
        for batch, data in enumerate(predict_dataloader):
            X = data['x'].to(config["DEVICE"])
            valid_mask = data['valid_mask'].to(config["DEVICE"])
            ego_windows = data['ego_windows'].to(config["DEVICE"])
            indices = data['index']
            
            # Generate ensemble predictions
            ensemble_predictions = []
            for _ in range(num_ensembles):
                if X.shape[1] == 50 and X.shape[2] == 50:
                    X = X.transpose(1, 2)
                predictions, _, _ = model_instance(X, valid_mask, ego_windows)
                ensemble_predictions.append(predictions)
            
            # Average ensemble predictions
            predictions = torch.stack(ensemble_predictions).mean(dim=0)
            
            # Convert predictions to original space
            pred_unnormalized = predict_dataset.getOriginalSpacePredictions(
                predictions.cpu().detach().numpy(),
                indices,
                indicator="prediction"
            )
            all_predictions.append(pred_unnormalized)

    all_np_predictions = np.concatenate(all_predictions, axis=0)[:, 1:, :2]
    
    # Ensure predictions have shape (2100, 60, 2)
    if all_np_predictions.shape[1] == 59:
        last_pred = all_np_predictions[:, -1:, :]
        all_np_predictions = np.concatenate([all_np_predictions, last_pred], axis=1)
    
    # Apply smoothing
    print("Applying smoothing...")
    test_data = np.load('test.npz')['data']
    last_pos = test_data[:, 0, 49, :2]  # Last known position
    const_vel = test_data[:, 0, 49, 2:4]  # Last known velocity
    
    # Calculate constant velocity based positions
    cvp = constantVelocityBasedPositions(last_pos, const_vel)
    
    smoothed_predictions = smooth_predictions(
        all_np_predictions, 
        const_vel,
        cvp,
        alpha=0.03,
        max_vel_change=0.3,
        confidence_threshold=0.9
    )
    
    # Save predictions
    print("Saving predictions...")
    pred_output = smoothed_predictions.reshape(-1, 2)
    output_df = pd.DataFrame(pred_output, columns=["x", "y"])
    output_df.index.name = "index"
    
    # Create Submissions directory if it doesn't exist
    os.makedirs(utils.SUBMISSION_DIR, exist_ok=True)
    output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "submission.csv"))
    print(f"Predictions saved to {os.path.join(utils.SUBMISSION_DIR, 'submission.csv')}")

def constantVelocityBasedPositions(last_pos, const_vel):
    """Calculate positions based on constant velocity assumption"""
    timesteps = np.arange(60).reshape(1, -1, 1)
    return last_pos.reshape(-1, 1, 2) + const_vel.reshape(-1, 1, 2) * timesteps

def smooth_predictions(predictions, const_vel, cvp, alpha=0.03, max_vel_change=0.3, confidence_threshold=0.9):
    """Apply adaptive temporal smoothing with velocity constraints"""
    smoothed = predictions.copy()
    batch_size, seq_len, _ = predictions.shape
    
    # Calculate initial deviation from constant velocity
    initial_deviation = np.linalg.norm(predictions - cvp, axis=2)
    confidence = np.exp(-initial_deviation)
    
    # Apply smoothing with adaptive weights
    for t in range(1, seq_len):
        # Calculate velocity
        vel = smoothed[:, t] - smoothed[:, t-1]
        vel_norm = np.linalg.norm(vel, axis=1)
        
        # Calculate adaptive smoothing factor
        smooth_factor = alpha * (1 - confidence[:, t])
        smooth_factor = np.clip(smooth_factor, 0, max_vel_change)
        
        # Apply smoothing
        smoothed[:, t] = smoothed[:, t-1] + vel * (1 - smooth_factor.reshape(-1, 1))
    
    return smoothed

if __name__ == "__main__":
    main() 