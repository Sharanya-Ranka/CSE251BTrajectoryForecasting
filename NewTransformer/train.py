import NewTransformer.data as dataset
from torch.utils.data import DataLoader
import NewTransformer.model as model
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import utilities as utils


class SimpleNNTrainer:
    def __init__(self, config):
        self.config = config
        self.best_eval_loss = float('inf')


    def performPipeline(self):
        self.setUpData()
        self.setUpModel()
        self.setUpOptimizer()
        training_progress = self.train()
        #self.predict()
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

    def setUpModel(self):
        self.model = model.EgoAgentNN(self.config)
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
        best_unnormalized_score = 15
        for epoch in range(self.config["EPOCHS"]):
            train_loss = self.train_epoch()
            eval_loss = self.eval_epoch()
            print(
                f"Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
            )
            training_progress[epoch] = {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }

            self.scheduler.step(eval_loss)
            print(
                f"Epoch {epoch+1}, Current Learning Rate: {self.scheduler.get_last_lr()[0]}"
            )

            if self.config.get("ANALYZE"):
                origspace_score_tr = self.analyze(self.train_dataloader)
                print(f"OrigSpace score (train) {origspace_score_tr}")

                origspace_score_te = self.analyze(self.test_dataloader)
                print(f"OrigSpace score (test) {origspace_score_te}")

                if origspace_score_te < best_unnormalized_score:
                    best_unnormalized_score = origspace_score_te
                    self.predict()

        return training_progress
    
    def weighted_mse_loss(self, pred, target, weights):
        diff = (pred-target) **2
        weighted_diff = diff*weights
        return weighted_diff.mean()

    def computeLoss(self, true, prediction):
        # print(
        #     f"computeLoss: prediction_shape={prediction.shape}, true_shape={true.shape}"
        # )
        # breakpoint()
        # loss = torch.mul(
        #     100, torch.mean((true[:, :, :2] - prediction[:, :, :2]) ** 2)
        # )  # + torch.mul(5, torch.mean((true[:, :, 2:5] - prediction[:, :, 2:5]) ** 2))
        weights = torch.tensor([2.0, 2.0, 0.75, 0.75, 1.0], device=true.device)
        loss = self.weighted_mse_loss(prediction, true, weights)
        return loss

    # Training function
    def train_epoch(self):
        num_batches = len(self.train_dataloader)

        self.model.train()

        train_loss, total = 0, 0
        for batch, (X, y, _) in enumerate(self.train_dataloader):
            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

            #Xf = torch.flatten(X, start_dim=1)
            # Compute prediction error
            prediction = self.model(X)

            puf = prediction.reshape(y.shape)
            loss = self.computeLoss(y, puf)
            # breakpoint()
            train_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print(f"Completed batch {batch} of {num_batches}")

        average_train_loss = float(train_loss / num_batches)
        return average_train_loss

    def eval_epoch(self):
        # Set the model to evaluation mode.
        # This disables dropout and uses accumulated batch norm statistics (if any).
        self.model.eval()

        eval_loss = 0
        num_batches = len(self.test_dataloader)

        # Disable gradient computation.
        # This reduces memory usage and speeds up computation.
        with torch.no_grad():
            for X, y, _ in self.test_dataloader:
                # Move data to the specified device
                X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

                #Xf = torch.flatten(X, start_dim=1)
                # Compute prediction error
                prediction = self.model(X)

                puf = prediction.reshape(y.shape)
                loss = self.computeLoss(y, puf)
                eval_loss += loss.item()

        # Calculate average loss
        average_eval_loss = float(eval_loss / num_batches)

        return average_eval_loss

    def analyze(self, dataloader: DataLoader):
        num_batches = self.config["ANALYZE_NUM_EXAMPLES"] / self.config["BATCH_SIZE"]

        self.model.eval()
        inference_steps = 60

        total_unnormalized = 0
        for batch, (X, y, org_indices) in enumerate(dataloader):
            # breakpoint()
            # Compute prediction error
            if batch > num_batches:
                break

            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

            #Xf = torch.flatten(X, start_dim=1)
            # Compute prediction error
            prediction = self.model(X)

            puf = prediction.reshape(y.shape)

            true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                y.detach().numpy(), org_indices
            )
            pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                puf.detach().numpy(), org_indices
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
                #Xf = torch.flatten(X, start_dim=1)
                
                # Get initial prediction
                predictions = self.model(X)
                predictions = predictions.reshape(-1, 60, 5)
                
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
        
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "simple_nn_submission.csv"))

def main():
    config = {
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 100,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.2,
        "NUM_SAMPLES": 10000,
        # Transformer specific parameters
        "D_INPUT": 1 * 50 * 5,
        "D_OUTPUT": 1 * 60 * 5,
        "D_HIDDEN": 5 * 50 * 5,
        "NUM_HIDDEN_LAYERS": 6,
        "NUM_QUERIES" : 2,
        #"DROPOUT": 0.3,  # Example dropout rate
        # Analysis parameters (optional based on ANALYZE flag)
        "ANALYZE": True,
        "ANALYZE_NUM_EXAMPLES": 100,
        "D_MODEL": 128,              # Model dimension
        "NHEAD": 8,                  # Number of attention heads
        "NUM_ENCODER_LAYERS": 6,     # Number of encoder layers
        "NUM_DECODER_LAYERS": 6,     # Number of decoder layers  
        "DIM_FEEDFORWARD": 512,      # Feedforward dimension
        "DROPOUT": 0.1,              # Dropout rate
    }

    trainer = SimpleNNTrainer(config)

    training_progress = trainer.performPipeline()
    df_progress = pd.DataFrame.from_dict(training_progress, orient="index")

    # Plotting
    plt.figure(figsize=(10, 6))
    df_progress[["train_loss", "eval_loss"]].plot()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")  # Optional
    plt.show()

# class LoRALayer(nn.Module):
#     """Manual implementation of LoRA layer"""
#     def __init__(self, original_layer, r=16, alpha=32, dropout=0.1):
#         super().__init__()
#         self.original_layer = original_layer
#         self.r = r
#         self.alpha = alpha
        
#         # Freeze original layer
#         for param in self.original_layer.parameters():
#             param.requires_grad = False
            
#         # Get dimensions
#         if isinstance(original_layer, nn.Linear):
#             in_features = original_layer.in_features
#             out_features = original_layer.out_features
#         else:
#             raise ValueError("LoRA currently only supports Linear layers")
            
#         # LoRA matrices
#         self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
#         self.lora_B = nn.Parameter(torch.zeros(out_features, r))
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         # Original output
#         original_output = self.original_layer(x)
        
#         # LoRA adaptation
#         lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
#         lora_output = lora_output * (self.alpha / self.r)
        
#         return original_output + lora_output
#     @property
#     def weight(self):
#         return self.original_layer.weight

#     @property
#     def bias(self):
#         return self.original_layer.bias


# class QLoRATrainer:
#     def __init__(self, config):
#         self.config = config
#         self.best_eval_loss = float('inf')

#     def apply_lora_to_model(self, model):
#         """Apply LoRA to transformer layers"""
#         lora_r = self.config.get("LORA_R", 16)
#         lora_alpha = self.config.get("LORA_ALPHA", 32)
#         lora_dropout = self.config.get("LORA_DROPOUT", 0.1)
        
#         # Apply LoRA to encoder layers
#         for i, layer in enumerate(model.transformer.encoder.layers):
#             # Replace linear layers in feedforward network
#             layer.linear1 = LoRALayer(layer.linear1, lora_r, lora_alpha, lora_dropout)
#             layer.linear2 = LoRALayer(layer.linear2, lora_r, lora_alpha, lora_dropout)
            
#             # For attention layers, we need to be more careful since they use MultiheadAttention
#             # We'll apply LoRA to the output projection of self-attention
#             if hasattr(layer.self_attn, 'out_proj'):
#                 layer.self_attn.out_proj = LoRALayer(layer.self_attn.out_proj, lora_r, lora_alpha, lora_dropout)
        
#         # Apply LoRA to decoder layers
#         for i, layer in enumerate(model.transformer.decoder.layers):
#             # Replace linear layers in feedforward network
#             layer.linear1 = LoRALayer(layer.linear1, lora_r, lora_alpha, lora_dropout)
#             layer.linear2 = LoRALayer(layer.linear2, lora_r, lora_alpha, lora_dropout)
            
#             # Apply LoRA to attention output projections
#             if hasattr(layer.self_attn, 'out_proj'):
#                 layer.self_attn.out_proj = LoRALayer(layer.self_attn.out_proj, lora_r, lora_alpha, lora_dropout)
#             if hasattr(layer.multihead_attn, 'out_proj'):
#                 layer.multihead_attn.out_proj = LoRALayer(layer.multihead_attn.out_proj, lora_r, lora_alpha, lora_dropout)
        
#         # Optionally apply LoRA to projection layers
#         if self.config.get("LORA_TARGET_PROJECTIONS", True):
#             model.input_projection = LoRALayer(model.input_projection, lora_r, lora_alpha, lora_dropout)
#             model.output_projection = LoRALayer(model.output_projection, lora_r, lora_alpha, lora_dropout)
        
#         return model

#     def performPipeline(self):
#         self.setUpData()
#         self.setUpModel()
#         self.setUpOptimizer()
#         training_progress = self.train()
#         return training_progress

#     def setUpData(self):
#         total_indices = np.arange(self.config["NUM_SAMPLES"])
#         train_indices, test_indices = train_test_split(
#             total_indices, test_size=self.config["TEST_SIZE"], random_state=42
#         )

#         train_config = {
#             "INDICES": train_indices,
#             "DATA_FILENAME": "train.npz",
#         }
#         test_config = {"INDICES": test_indices, "DATA_FILENAME": "train.npz"}

#         self.train_dataset = dataset.EgoAgentNormalizedDataset(train_config)
#         self.test_dataset = dataset.EgoAgentNormalizedDataset(test_config)

#         self.train_dataloader = DataLoader(
#             self.train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
#         )
#         self.test_dataloader = DataLoader(
#             self.test_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
#         )

#         # For predictions
#         predict_indices = np.arange(2100)

#         predict_config = {
#             "INDICES": predict_indices,
#             "DATA_FILENAME": "test.npz",
#             "INFERENCE": True,
#         }

#         self.predict_dataset = dataset.EgoAgentNormalizedDataset(predict_config)

#         self.predict_dataloader = DataLoader(
#             self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
#         )

#     def setUpModel(self):
#         # Initialize base model
#         self.model = model.EgoAgentTransformer(self.config)
#         print("Base model:")
#         print(self.model)
        
#         # Apply QLoRA if enabled
#         if self.config.get("USE_QLORA", True):
#             print("\nApplying LoRA...")
#             self.model = self.apply_lora_to_model(self.model)
#             #self.model = self.model.to(self.config["DEVICE"])
            
#             # Print parameter statistics
#             trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#             total_params = sum(p.numel() for p in self.model.parameters())
#             print(f"\nLoRA Parameter Statistics:")
#             print(f"Trainable parameters: {trainable_params:,}")
#             print(f"Total parameters: {total_params:,}")
#             print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
            
#             # Print which parameters are trainable
#             print("\nTrainable modules:")
#             for name, param in self.model.named_parameters():
#                 if param.requires_grad:
#                     print(f"  {name}: {param.shape}")

#     def setUpOptimizer(self):
#         # Only optimize parameters that require gradients (LoRA parameters)
#         trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
#         self.optimizer = torch.optim.AdamW(
#             trainable_params, 
#             lr=self.config["LEARNING_RATE"],
#             weight_decay=self.config.get("WEIGHT_DECAY", 0.01)
#         )

#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, patience=5, factor=0.2
#         )

#     def train(self):
#         training_progress = {}
#         best_unnormalized_score = 15
        
#         for epoch in range(self.config["EPOCHS"]):
#             train_loss = self.train_epoch()
#             eval_loss = self.eval_epoch()
#             print(
#                 f"Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
#             )
#             training_progress[epoch] = {
#                 "train_loss": train_loss,
#                 "eval_loss": eval_loss,
#             }

#             self.scheduler.step(eval_loss)
#             print(
#                 f"Epoch {epoch+1}, Current Learning Rate: {self.scheduler.get_last_lr()[0]}"
#             )

#             if self.config.get("ANALYZE"):
#                 origspace_score_tr = self.analyze(self.train_dataloader)
#                 print(f"OrigSpace score (train) {origspace_score_tr}")

#                 origspace_score_te = self.analyze(self.test_dataloader)
#                 print(f"OrigSpace score (test) {origspace_score_te}")

#                 if origspace_score_te < best_unnormalized_score:
#                     best_unnormalized_score = origspace_score_te
#                     self.predict()
#                     # Save best model
#                     if self.config.get("USE_QLORA", True):
#                         self.save_lora_model(f"best_lora_epoch_{epoch}.pth")

#         return training_progress
    
#     def weighted_mse_loss(self, pred, target, weights):
#         diff = (pred - target) ** 2
#         weighted_diff = diff * weights
#         return weighted_diff.mean()

#     def computeLoss(self, true, prediction):
#         weights = torch.tensor([2.0, 2.0, 0.75, 0.75, 1.0], device=true.device)
#         loss = self.weighted_mse_loss(prediction, true, weights)
#         return loss

#     def train_epoch(self):
#         num_batches = len(self.train_dataloader)
#         self.model.train()

#         train_loss, total = 0, 0
#         for batch, (X, y, _) in enumerate(self.train_dataloader):
#             X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

#             # Compute prediction error
#             prediction = self.model(X)
#             puf = prediction.reshape(y.shape)
#             loss = self.computeLoss(y, puf)

#             train_loss += loss.item()

#             # Backpropagation
#             self.optimizer.zero_grad()
#             loss.backward()
            
#             # Gradient clipping
#             if self.config.get("GRAD_CLIP"):
#                 torch.nn.utils.clip_grad_norm_(
#                     self.model.parameters(), 
#                     self.config["GRAD_CLIP"]
#                 )
            
#             self.optimizer.step()

#         average_train_loss = float(train_loss / num_batches)
#         return average_train_loss

#     def eval_epoch(self):
#         self.model.eval()
#         eval_loss = 0
#         num_batches = len(self.test_dataloader)

#         with torch.no_grad():
#             for X, y, _ in self.test_dataloader:
#                 X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

#                 prediction = self.model(X)
#                 puf = prediction.reshape(y.shape)
#                 loss = self.computeLoss(y, puf)
#                 eval_loss += loss.item()

#         average_eval_loss = float(eval_loss / num_batches)
#         return average_eval_loss

#     def analyze(self, dataloader):
#         num_batches = self.config["ANALYZE_NUM_EXAMPLES"] / self.config["BATCH_SIZE"]
#         self.model.eval()

#         total_unnormalized = 0
#         for batch, (X, y, org_indices) in enumerate(dataloader):
#             if batch > num_batches:
#                 break

#             X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

#             prediction = self.model(X)
#             puf = prediction.reshape(y.shape)

#             true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
#                 y.detach().cpu().numpy(), org_indices
#             )
#             pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
#                 puf.detach().cpu().numpy(), org_indices
#             )
#             unnormalized_metric = dataloader.dataset.computeOriginalSpaceMetric(
#                 true_unnormalized, pred_unnormalized
#             )
#             total_unnormalized += unnormalized_metric

#         return total_unnormalized / min(num_batches, len(dataloader))

#     def predict(self):
#         all_predictions = []
#         dataloader = self.predict_dataloader

#         self.model.eval()
#         with torch.no_grad():
#             for batch, (X, _, org_indices) in enumerate(dataloader):
#                 X = X.to(self.config["DEVICE"])
                
#                 predictions = self.model(X)
#                 predictions = predictions.reshape(-1, 60, 5)
                
#                 pred_np = predictions.cpu().numpy()
#                 pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(pred_np, org_indices)
#                 all_predictions.append(pred_unnormalized)

#         all_np_predictions = np.concatenate(all_predictions, axis=0)[:, :, :2]
#         self.convertAndSavePredictions(all_np_predictions)

#     def convertAndSavePredictions(self, predictions):
#         assert tuple(predictions.shape) == (2100, 60, 2)

#         pred_output = predictions.reshape(-1, 2)
#         output_df = pd.DataFrame(pred_output, columns=["x", "y"])
#         output_df.index.name = "index"
        
#         os.makedirs(utils.SUBMISSION_DIR, exist_ok=True)
#         output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "lora_transformer_submission.csv"))

#     def save_lora_model(self, filepath):
#         """Save only the LoRA parameters"""
#         lora_state_dict = {}
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
#                 lora_state_dict[name] = param.data.clone()
        
#         torch.save({
#             'lora_state_dict': lora_state_dict,
#             'config': self.config
#         }, filepath)
#         print(f"LoRA model saved to {filepath}")

#     def load_lora_model(self, filepath):
#         """Load LoRA parameters"""
#         checkpoint = torch.load(filepath, map_location=self.config["DEVICE"])
#         lora_state_dict = checkpoint['lora_state_dict']
        
#         # Load LoRA parameters
#         model_dict = self.model.state_dict()
#         model_dict.update(lora_state_dict)
#         self.model.load_state_dict(model_dict)
#         print(f"LoRA model loaded from {filepath}")

# def main():
#     config = {
#         "BATCH_SIZE": 32,  # Reduced for memory efficiency with QLoRA
#         "LEARNING_RATE": 2e-4,  # Lower learning rate typical for LoRA
#         "EPOCHS": 100,
#         "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
#         "TEST_SIZE": 0.2,
#         "NUM_SAMPLES": 10000,
        
#         # Transformer specific parameters
#         "D_MODEL": 256,              # Model dimension
#         "NHEAD": 8,                  # Number of attention heads
#         "NUM_ENCODER_LAYERS": 6,     # Number of encoder layers
#         "NUM_DECODER_LAYERS": 6,     # Number of decoder layers  
#         "DIM_FEEDFORWARD": 1024,     # Feedforward dimension
#         "DROPOUT": 0.1,              # Dropout rate
        
#         # LoRA specific parameters
#         "USE_QLORA": True,           # Enable LoRA
#         "LORA_R": 16,                # LoRA rank
#         "LORA_ALPHA": 32,            # LoRA alpha (scaling factor)
#         "LORA_DROPOUT": 0.1,         # LoRA dropout
#         "LORA_TARGET_PROJECTIONS": True,  # Whether to target projection layers
#         "WEIGHT_DECAY": 0.01,        # Weight decay for AdamW
#         "GRAD_CLIP": 1.0,            # Gradient clipping
        
#         # Analysis parameters
#         "ANALYZE": True,
#         "ANALYZE_NUM_EXAMPLES": 100,
#     }

#     trainer = QLoRATrainer(config)
#     training_progress = trainer.performPipeline()
    
#     df_progress = pd.DataFrame.from_dict(training_progress, orient="index")

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     df_progress[["train_loss", "eval_loss"]].plot()
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("LoRA Training and Validation Loss over Epochs")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("lora_loss_curve.png")
#     plt.show()


