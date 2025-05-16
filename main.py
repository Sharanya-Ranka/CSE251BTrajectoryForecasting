import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

import SimpleTransformer.train as st_run
import SimpleTransformer.data_create as dc_run
import SimpleNeuralNetwork.train as sn_run
import ConstantVelocityPlusNN.data_create as cvnn_dc_run
import ConstantVelocityPlusNN.train as cvnn_run
import LSTMWeightBasedLoss.data_create as lswbdc_run 
from LSTMWeightBasedLoss.model import EgoAgentNN
import torch
import numpy as np
import pandas as pd
from LSTMWeightBasedLoss.data_create import createUnnormalizedY, createOriginalSpacePredictions

# First run data creation for ConstantVelocityPlusNN
cvnn_dc_run.main()

# Then run data creation for LSTMWeightBasedLoss
lswbdc_run.main()

# Then run training
config = {
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.001,
    "EPOCHS": 100,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "TEST_SIZE": 0.2,
    "NUM_SAMPLES": 10000,
    "DROPOUT": 0.3,
    "ANALYZE": True,
    "ANALYZE_NUM_EXAMPLES": 100,
}

# Create model
model = EgoAgentNN(config).to(config["DEVICE"])

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

# Load data
train_file = np.load(os.path.join("LSTMWeightBasedLoss", "Data", "IntermediateData", "LSTMWeightBasedLoss", "train.npz"))
test_file = np.load(os.path.join("LSTMWeightBasedLoss", "Data", "IntermediateData", "LSTMWeightBasedLoss", "test.npz"))

trainX = torch.tensor(train_file["X"], dtype=torch.float32)
trainY = torch.tensor(train_file["Y"], dtype=torch.float32)
testX = torch.tensor(test_file["X"], dtype=torch.float32)

# Store original data for analysis
train_data = train_file["data"]
test_data = test_file["data"]

# Create datasets and dataloaders
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(trainX, trainY)
train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)

test_dataset = TensorDataset(testX)
test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

def analyze_predictions(model, dataloader, original_data, num_examples=100, is_test=False):
    model.eval()
    total_orig_space_score = 0
    count = 0
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if count >= num_examples and not is_test:
                break
                
            if is_test:
                batch_X = batch[0]
                predictions = model(batch_X.to(config["DEVICE"]))
                predictions_np = predictions.cpu().numpy()
                
                # Convert to original space
                unnorm_predictions = createUnnormalizedY(predictions_np)
                batch_original_data = original_data[batch_idx * config["BATCH_SIZE"]:(batch_idx + 1) * config["BATCH_SIZE"]]
                orig_space_preds = createOriginalSpacePredictions(unnorm_predictions, batch_original_data)
                
                # Store predictions
                all_predictions.append(orig_space_preds)
                
                # For test data, we can't compute score since we don't have ground truth
                print("Test predictions shape:", orig_space_preds.shape)
                count += len(batch_X)
            else:
                batch_X, batch_Y = batch
                batch_X, batch_Y = batch_X.to(config["DEVICE"]), batch_Y.to(config["DEVICE"])
                predictions = model(batch_X)
                
                # Convert to numpy for analysis
                predictions_np = predictions.cpu().numpy()
                true_np = batch_Y.cpu().numpy()
                
                # Unnormalize predictions
                unnorm_predictions = createUnnormalizedY(predictions_np)
                unnorm_true = createUnnormalizedY(true_np)
                
                # Get original space predictions
                batch_original_data = original_data[batch_idx * config["BATCH_SIZE"]:(batch_idx + 1) * config["BATCH_SIZE"]]
                orig_space_preds = createOriginalSpacePredictions(unnorm_predictions, batch_original_data)
                orig_space_true = createOriginalSpacePredictions(unnorm_true, batch_original_data)
                
                # Calculate MSE in original space
                mse = np.mean((orig_space_preds - orig_space_true) ** 2)
                total_orig_space_score += mse
                count += len(batch_X)
    
    if is_test:
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Only keep x and y coordinates
        all_predictions = all_predictions[:, :, :2]
        
        # Reshape to (2100*60, 2) for DataFrame
        pred_output = all_predictions.reshape(-1, 2)
        
        # Create DataFrame with just x and y
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])
        output_df.index.name = "index"
        
        # Save to CSV
        output_dir = os.path.join("Submissions")
        os.makedirs(output_dir, exist_ok=True)
        output_df.to_csv(os.path.join(output_dir, "lstm_submission.csv"))
        print(f"Saved predictions to {os.path.join(output_dir, 'lstm_submission.csv')}")
        return None  # Return None for test data since we can't compute score
    
    if count == 0:
        return None
    return total_orig_space_score / count

# Training loop
best_loss = float('inf')
best_orig_space_score = float('inf')
for epoch in range(config["EPOCHS"]):
    model.train()
    total_loss = 0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(config["DEVICE"]), batch_Y.to(config["DEVICE"])
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = torch.nn.functional.mse_loss(outputs, batch_Y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{config['EPOCHS']}, Loss: {avg_loss:.6f}")
    
    # Save model if we get a lower loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_lstm_model_loss.pth")
        print(f"Saved new best model with loss: {best_loss:.6f}")
    
    # Analyze if needed
    if config["ANALYZE"] and (epoch + 1) % 10 == 0:
        # Analyze training set
        train_orig_score = analyze_predictions(model, train_loader, train_data, config["ANALYZE_NUM_EXAMPLES"])
        print(f"Training Original Space Score: {train_orig_score:.6f}")
        
        # Analyze test set - we can't compute score for test data
        print("Generating test predictions...")
        analyze_predictions(model, test_loader, test_data, config["ANALYZE_NUM_EXAMPLES"], is_test=True)
        print("Test Original Space Score: N/A (no ground truth available)")
        
        # Save model if we get a better original space score
        if train_orig_score < best_orig_space_score:
            best_orig_space_score = train_orig_score
            torch.save(model.state_dict(), "best_lstm_model_score.pth")
            print(f"Saved new best model with score: {best_orig_space_score:.6f}")
            
            # Generate predictions for the best model
            print("\nGenerating predictions for best model...")
            analyze_predictions(model, test_loader, test_data, is_test=True)

# After training, save final predictions
print("\nGenerating final predictions...")
analyze_predictions(model, test_loader, test_data, is_test=True)
