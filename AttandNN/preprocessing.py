import numpy as np
import os
import pandas as pd
from utilities import *

VEL_MULTIPLIER = 1 / 5
HEAD_MULTIPLIER = 1 / np.pi
POS_DIFF_MULTIPLIER = 1 / 50
POS_MULTIPLIER = 1 / 250
TYPE_MULTIPLIER = 1 / 10
VHID_MULTIPLIER = 1 / 50
TS_MULTIPLIER = 1 / 110

def viewPercentiles(data, percentile_interval=5):
    percentiles = [0.1, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.9]
    p_values = np.percentile(data, percentiles)
    for p, value in zip(percentiles, p_values):
        print(f"{p}th percentile: {value}")

def constantVelocityBasedPositions(last_pos, const_vel):
    pred_pos = np.zeros((last_pos.shape[0], 60, 2))
    for t in range(60):
        pred_pos[:, t] = last_pos + np.multiply(const_vel, 0.1) * t
    return pred_pos

def addAuxiliaryData(dataset):
    req_shape = (dataset.shape[0], dataset.shape[1], dataset.shape[2], 1)
    valid = np.where(
        (dataset[:, :, :, 0] == 0) & (dataset[:, :, :, 1] == 0), 0, 1
    ).reshape(req_shape)
    vh_id = np.broadcast_to(np.arange(50).reshape(1, 50, 1, 1), req_shape)
    ts = np.broadcast_to(np.arange(dataset.shape[2]).reshape(1, 1, -1, 1), req_shape)
    
    aux_dataset = np.concatenate([dataset, valid, vh_id, ts], axis=-1)
    return aux_dataset

def createXFeatures(dataset):
    norm_pos = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., :2] - dataset[:, 0:1, 49:50, :2]) * POS_MULTIPLIER,
    )
    norm_vel = np.where(
        (dataset[:, :, :, 6:7] == 0),
        0,
        (dataset[..., 2:4]) * VEL_MULTIPLIER,
    )
    norm_head = np.where(
        (dataset[:, :, :, 6:7] == 0), 0, (dataset[..., 4:5]) * HEAD_MULTIPLIER
    )
    norm_types = np.where(
        (dataset[:, :, :, 6:7] == 0), -1, (10 - dataset[..., 5:6]) * TYPE_MULTIPLIER
    )
    norm_valid = dataset[..., 6:7]
    norm_vh_id = dataset[..., 7:8] * VHID_MULTIPLIER
    norm_ts = dataset[..., 8:9] * TS_MULTIPLIER
    
    norm_dataset = np.concatenate(
        [norm_pos, norm_vel, norm_head, norm_types, norm_valid, norm_vh_id, norm_ts],
        axis=-1,
    )
    return norm_dataset[:, :, :50, :]

def createYTarget(dataset):
    cvp = constantVelocityBasedPositions(dataset[:, 0, 49, 0:2], dataset[:, 0, 49, 2:4])
    to_predict_pos = (dataset[:, 0, 50:, :2] - cvp) * POS_DIFF_MULTIPLIER
    return to_predict_pos

def createOriginalSpacePredictions(Y, org_data):
    """
    Enhanced to handle both single and multi-modal predictions
    Y: can be (batch, 60, 2) or (batch, num_modes, 60, 2)
    """
    cvp = constantVelocityBasedPositions(org_data[:, 0, 49, 0:2], org_data[:, 0, 49, 2:4])
    
    # Handle multi-modal predictions
    if len(Y.shape) == 4:  # Multi-modal: (batch, modes, timesteps, coords)
        batch_size, num_modes, timesteps, coords = Y.shape
        scaledY = Y / POS_DIFF_MULTIPLIER
        
        # Expand cvp to match multi-modal shape
        cvp_expanded = np.expand_dims(cvp, axis=1)  # (batch, 1, timesteps, coords)
        cvp_expanded = np.broadcast_to(cvp_expanded, (batch_size, num_modes, timesteps, coords))
        
        orig_space_pos = cvp_expanded + scaledY
    else:  # Single mode: (batch, timesteps, coords)
        scaledY = Y / POS_DIFF_MULTIPLIER
        orig_space_pos = cvp + scaledY
    
    return orig_space_pos

def extractAgentTypes(dataset):
    """
    Extract agent types for each agent at each timestep
    Returns: (batch_size, num_agents, max_timesteps) with agent type IDs
    """
    # Agent types are in dataset[..., 5] (before normalization)
    # 0=unknown, 1=vehicle, 2=pedestrian, 3=cyclist, etc.
    agent_types = dataset[..., 5].astype(np.int32)
    return agent_types[:, :, :50]  # Only first 50 timesteps

def extractLaneData(dataset=None):
    """
    Placeholder for lane/map data extraction
    In Argoverse 2, you would extract lane centerlines, traffic lights, etc.
    For now, returns None - implement based on your data structure
    """
    # TODO: Implement lane data extraction if available
    # This would typically involve:
    # - Loading map data
    # - Finding relevant lanes around each trajectory
    # - Encoding lane centerlines as sequences of (x, y) coordinates
    return None

def computeOriginalSpaceMetric(true_orig, pred_orig):
    """
    Enhanced to handle multi-modal predictions
    """
    if len(pred_orig.shape) == 4:  # Multi-modal predictions
        # For multi-modal, compute metric for best mode
        batch_size, num_modes = pred_orig.shape[:2]
        
        # Expand true trajectories to match multi-modal shape
        true_expanded = np.expand_dims(true_orig, axis=1)
        true_expanded = np.broadcast_to(true_expanded, pred_orig.shape)
        
        # Compute L2 distances for each mode
        distances = np.linalg.norm(pred_orig - true_expanded, axis=-1)  # (batch, modes, timesteps)
        mode_errors = np.mean(distances, axis=-1)  # Average over timesteps
        
        # Use best mode (minimum error) for each sample
        best_mode_errors = np.min(mode_errors, axis=1)  # (batch,)
        return np.mean(best_mode_errors)
    else:
        # Single mode prediction
        distances = np.linalg.norm(pred_orig - true_orig, axis=-1)
        return np.mean(distances)

def createDataAugmentation(X, Y, aug_factor=0.1):
    """
    Simple data augmentation for motion forecasting
    """
    batch_size = X.shape[0]
    
    # Random rotation (small angles)
    angles = np.random.normal(0, 0.1, batch_size)  # Small rotation angles
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # Apply rotation to position and velocity features
    X_aug = X.copy()
    Y_aug = Y.copy()
    
    for i in range(batch_size):
        # Rotate positions (features 0,1)
        pos_x = X[i, :, :, 0] * cos_angles[i] - X[i, :, :, 1] * sin_angles[i]
        pos_y = X[i, :, :, 0] * sin_angles[i] + X[i, :, :, 1] * cos_angles[i]
        X_aug[i, :, :, 0] = pos_x
        X_aug[i, :, :, 1] = pos_y
        
        # Rotate velocities (features 2,3)
        vel_x = X[i, :, :, 2] * cos_angles[i] - X[i, :, :, 3] * sin_angles[i]
        vel_y = X[i, :, :, 2] * sin_angles[i] + X[i, :, :, 3] * cos_angles[i]
        X_aug[i, :, :, 2] = vel_x
        X_aug[i, :, :, 3] = vel_y
        
        # Rotate target trajectories
        traj_x = Y[i, :, 0] * cos_angles[i] - Y[i, :, 1] * sin_angles[i]
        traj_y = Y[i, :, 0] * sin_angles[i] + Y[i, :, 1] * cos_angles[i]
        Y_aug[i, :, 0] = traj_x
        Y_aug[i, :, 1] = traj_y
    
    return X_aug, Y_aug

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process training data
    train_file = np.load(os.path.join(script_dir, "train.npz"))
    train_data = train_file["data"]
    print("train_data's shape", train_data.shape)
    
    train_aux = addAuxiliaryData(train_data)
    trainX = createXFeatures(train_aux)
    trainY = createYTarget(train_aux)
    
    # Extract additional features for enhanced model
    train_agent_types = extractAgentTypes(train_aux)
    train_lane_data = extractLaneData(train_aux)  # Returns None for now
    
    print(f"Shape of trainX: {trainX.shape}")
    print(f"Shape of trainY: {trainY.shape}")
    print(f"Shape of train_agent_types: {train_agent_types.shape}")
    
    # Save training data with additional features in same directory
    save_dict = {
        "data": train_aux,
        "X": trainX,
        "Y": trainY,
        "agent_types": train_agent_types,
    }
    
    if train_lane_data is not None:
        save_dict["lane_data"] = train_lane_data
    
    np.savez(
        os.path.join(script_dir, "train_processed.npz"),
        **save_dict
    )
    
    # Process test data
    test_file = np.load(os.path.join(script_dir, "test_input.npz"))
    test_data = test_file["data"]
    print("test_data's shape", test_data.shape)
    
    test_aux = addAuxiliaryData(test_data)
    testX = createXFeatures(test_aux)
    test_agent_types = extractAgentTypes(test_aux)
    test_lane_data = extractLaneData(test_aux)  # Returns None for now
    
    print(f"Shape of testX: {testX.shape}")
    print(f"Shape of test_agent_types: {test_agent_types.shape}")
    
    # Save test data with additional features in same directory
    test_save_dict = {
        "data": test_aux,
        "X": testX,
        "agent_types": test_agent_types,
    }
    
    if test_lane_data is not None:
        test_save_dict["lane_data"] = test_lane_data
    
    np.savez(
        os.path.join(script_dir, "test_processed.npz"),
        **test_save_dict
    )
    
    print("Data preprocessing completed with enhanced features!")
    print(f"Files saved in: {script_dir}")

if __name__ == "__main__":
    main()