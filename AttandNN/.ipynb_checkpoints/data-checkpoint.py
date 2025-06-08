import torch
from torch.utils.data import Dataset
import numpy as np
import os
from utilities import *

# Import your preprocessing functions
VEL_MULTIPLIER = 1 / 5
HEAD_MULTIPLIER = 1 / np.pi
POS_DIFF_MULTIPLIER = 1 / 50
POS_MULTIPLIER = 1 / 250

def constantVelocityBasedPositions(last_pos, const_vel):
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    for t in range(60):
        pred_pos[:, t] = last_pos + np.multiply(const_vel, 0.1) * t
    return pred_pos

# Import the exact functions from your preprocessing file
from preprocessing import createOriginalSpacePredictions, computeOriginalSpaceMetric

class AllAgentsNormalizedDataset(Dataset):
    def __init__(self, config):
        """
        config should contain:
        - INDICES: list of indices to use
        - DATA_FILENAME: name of the .npz file (e.g., "train.npz" or "test.npz")
        - INFERENCE: (optional) boolean indicating if this is for inference
        """
        self.config = config
        self.indices = config["INDICES"]
        self.data_filename = config["DATA_FILENAME"]
        self.inference = config.get("INFERENCE", False)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load the preprocessed data"""
        data_path = os.path.join(DATA_DIR, "IntermediateData", "AttentionAndNN", self.data_filename)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        data_file = np.load(data_path)
        
        # Load features and targets
        self.X = data_file["X"]  # Input features
        self.original_data = data_file["data"]  # Original data for denormalization
        
        # Load targets if not inference
        if not self.inference:
            self.Y = data_file["Y"]  # Target trajectories
        else:
            self.Y = None
            
        # Load additional features if available
        self.agent_types = data_file.get("agent_types", None)
        self.lane_data = data_file.get("lane_data", None)
        
        # Filter by indices
        self.X = self.X[self.indices]
        self.original_data = self.original_data[self.indices]
        
        if self.Y is not None:
            self.Y = self.Y[self.indices]
            
        if self.agent_types is not None:
            self.agent_types = self.agent_types[self.indices]
            
        if self.lane_data is not None:
            self.lane_data = self.lane_data[self.indices]
            
        print(f"Loaded dataset with {len(self.indices)} samples")
        print(f"X shape: {self.X.shape}")
        if self.Y is not None:
            print(f"Y shape: {self.Y.shape}")
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
        - X: input features (num_agents, timesteps, features)
        - Y: target trajectory (timesteps, coords) - only if not inference
        - index: original index for this sample
        """
        original_idx = self.indices[idx]
        
        x = torch.FloatTensor(self.X[idx])
        
        if self.inference:
            # For inference, return only input and index
            return x, None, original_idx
        else:
            y = torch.FloatTensor(self.Y[idx])
            return x, y, original_idx
    
    def getOriginalSpacePredictions(self, predictions, org_indices, indicator="prediction"):
        """
        Convert normalized predictions back to original space
        
        Args:
            predictions: model predictions in normalized space
            org_indices: original indices for the batch
            indicator: "prediction" or "true" for logging
            
        Returns:
            predictions in original coordinate space
        """
        # Get original data for these indices
        batch_original_data = self.original_data[org_indices]
        
        # Convert to original space using your preprocessing function
        orig_space_preds = createOriginalSpacePredictions(predictions, batch_original_data)
        
        return orig_space_preds
    
    def unnormalizeData(self, normalized_data, org_indices):
        """
        Unnormalize data back to original space
        This is a wrapper around getOriginalSpacePredictions for compatibility
        """
        return self.getOriginalSpacePredictions(normalized_data, org_indices, "unnormalized")
    
    def computeOriginalSpaceMetric(self, true_orig, pred_orig):
        """
        Use the same function from your preprocessing file
        """
        return computeOriginalSpaceMetric(true_orig, pred_orig)


class EnhancedDataset(AllAgentsNormalizedDataset):
    """
    Enhanced dataset that provides additional features for the improved model
    """
    
    def __getitem__(self, idx):
        """
        Returns enhanced features including agent types and lane data
        """
        original_idx = self.indices[idx]
        
        x = torch.FloatTensor(self.X[idx])
        
        # Get additional features
        agent_types = None
        if self.agent_types is not None:
            agent_types = torch.LongTensor(self.agent_types[idx])
            
        lane_data = None
        if self.lane_data is not None:
            lane_data = torch.FloatTensor(self.lane_data[idx])
        
        if self.inference:
            return {
                'input': x,
                'agent_types': agent_types,
                'lane_data': lane_data,
                'index': original_idx
            }
        else:
            y = torch.FloatTensor(self.Y[idx])
            return {
                'input': x,
                'target': y,
                'agent_types': agent_types,
                'lane_data': lane_data,
                'index': original_idx
            }