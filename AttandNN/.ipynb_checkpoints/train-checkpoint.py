import AttandNN.data as dataset
from torch.utils.data import DataLoader
import AttandNN.model as model
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import utilities as utils


class AttentionAndNNTrainer:
    def __init__(self, config):
        self.config = config

    def performPipeline(self):
        self.setUpData()
        self.setUpModel()
        self.setUpOptimizer()
        self.train()
        # self.predict()

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
        
        # Initialize multi-modal loss
        self.multimodal_loss = model.MultiModalLoss(
            num_modes=self.config.get('NUM_MODES', 6),
            reduction='mean'
        )
        
        print(self.model)

    def setUpOptimizer(self):
        # Define the optimizer being used
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["LEARNING_RATE"]
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.1
        )

    def train(self):
        training_progress = {}
        # best_unnormalized_score = 15
        for epoch in range(self.config["EPOCHS"]):
            print(f"Performing epoch {epoch}")
            self.epoch = epoch
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

            if self.config.get("ANALYZE") and self.epoch % 5 == 0:
                origspace_score_tr = self.analyze(self.train_dataloader)
                print(f"OrigSpace score (train) {origspace_score_tr}")

                # origspace_score_te = self.analyze(self.test_dataloader)
                # print(f"OrigSpace score (test) {origspace_score_te}")

                # if unnormalized_score < best_unnormalized_score:
                #     best_unnormalized_score = unnormalized_score
                #     self.predict()

        return training_progress

    def computeLoss(self, true, pred_trajectories, mode_probs=None):
        """
        Enhanced loss function for multi-modal predictions
        true: (batch_size, 60, 2) - ground truth trajectories
        pred_trajectories: (batch_size, num_modes, 60, 2) - predicted trajectories
        mode_probs: (batch_size, num_modes) - mode probabilities
        """
        if mode_probs is not None:
            # Multi-modal loss
            loss = self.multimodal_loss(pred_trajectories, mode_probs, true)
        else:
            # Fallback to single-mode loss if model returns single prediction
            if len(pred_trajectories.shape) == 4:  # Multi-modal output
                # Use best mode based on minimum distance
                batch_size = pred_trajectories.size(0)
                num_modes = pred_trajectories.size(1)
                
                true_expanded = true.unsqueeze(1).expand(-1, num_modes, -1, -1)
                distances = torch.norm(pred_trajectories - true_expanded, dim=-1).mean(dim=-1)
                best_modes = torch.argmin(distances, dim=1)
                best_predictions = pred_trajectories[torch.arange(batch_size), best_modes]
                
                loss = 100 * torch.mean((true - best_predictions) ** 2)
            else:
                # Single mode prediction
                loss = 100 * torch.mean((true - pred_trajectories) ** 2)

        return loss

    def viewPercentilesWithSampleIndices(self, data, org_indices, percentile_interval=5):
        percentiles = [0.1, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.9]
        p_values = np.percentile(data, percentiles, method='nearest')

        for p, value in zip(percentiles, p_values):
            print(f"{p}th percentile: {value}")
            example_ind = np.where(value == data)[0][0]
            org_ind = org_indices[example_ind]
            print(f"In context indices\n{np.where(value == data)}")
            print(f"Original index example={org_ind}")

    # Training function
    def train_epoch(self):
        num_batches = len(self.train_dataloader)

        self.model.train()

        train_loss, total = 0, 0
        for batch, (X, Y, indices) in enumerate(self.train_dataloader):
            # print(f"On batch={batch}")
            X = X.to(self.config["DEVICE"])
            Y = Y.to(self.config["DEVICE"])
            
            # Model now returns trajectories and mode probabilities
            model_output = self.model(X)
            
            if isinstance(model_output, tuple):
                pred_trajectories, mode_probs = model_output
                loss = self.computeLoss(Y, pred_trajectories, mode_probs)
            else:
                # Backward compatibility for single output
                pred_trajectories = model_output
                loss = self.computeLoss(Y, pred_trajectories)
            
            train_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        average_train_loss = float(train_loss / num_batches)
        return average_train_loss

    def eval_epoch(self):
        # Set the model to evaluation mode.
        # This disables dropout and uses accumulated batch norm statistics (if any).
        self.model.eval()

        eval_loss = 0
        num_batches = len(self.test_dataloader)  # 3

        # Disable gradient computation.
        # This reduces memory usage and speeds up computation.
        with torch.no_grad():
            for batch, (X, Y, _) in enumerate(self.test_dataloader):
                # print(f"On batch={batch}")
                # if batch > 2:
                #     break
                X = X.to(self.config["DEVICE"])
                Y = Y.to(self.config["DEVICE"])

                model_output = self.model(X)
                
                if isinstance(model_output, tuple):
                    pred_trajectories, mode_probs = model_output
                    loss = self.computeLoss(Y, pred_trajectories, mode_probs)
                else:
                    pred_trajectories = model_output
                    loss = self.computeLoss(Y, pred_trajectories)

                # breakpoint()
                eval_loss += loss.item()

        # Calculate average loss
        average_eval_loss = float(eval_loss / num_batches)

        return average_eval_loss

    def analyze(self, dataloader: DataLoader):
        num_batches = self.config["ANALYZE_NUM_BATCHES"]

        self.model.eval()
        inference_steps = 60
        # torch.cuda.empty_cache()

        total_unnormalized = 0
        with torch.no_grad():
            for batch, (X, Y, org_indices) in enumerate(dataloader):
                # breakpoint()
                # Compute prediction error
                if batch >= num_batches:
                    break

                X = X.to(self.config["DEVICE"])
                Y = Y.to(self.config["DEVICE"])

                model_output = self.model(X)
                
                # Handle multi-modal output
                if isinstance(model_output, tuple):
                    pred_trajectories, mode_probs = model_output
                    # Use most confident prediction for analysis
                    best_mode_idx = torch.argmax(mode_probs, dim=1)
                    prediction = pred_trajectories[torch.arange(len(pred_trajectories)), best_mode_idx]
                else:
                    prediction = model_output

                true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    X[:, 0, 50:, :2].cpu().detach().numpy(), org_indices, indicator="true"
                )
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    prediction.cpu().detach().numpy(),
                    org_indices,
                    indicator="prediction",
                )
                unnormalized_metric = dataloader.dataset.computeOriginalSpaceMetric(
                    true_unnormalized, pred_unnormalized
                )
                if self.epoch > 10:
                    breakpoint()
                total_unnormalized += unnormalized_metric

        return total_unnormalized / min(num_batches, len(dataloader))

    def predict(self):
        all_predictions = []
        all_mode_probs = []  # Store mode probabilities for multi-modal outputs
        dataloader = self.predict_dataloader
        inference_steps = 60

        self.model.eval()

        total_unnormalized = 0
        for batch, (X, y, org_indices) in enumerate(dataloader):

            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
            
            # Note: This prediction loop seems to be for autoregressive generation
            # You might need to modify this based on your specific use case
            for i in range(inference_steps):
                model_output = self.model(X, y)
                
                if isinstance(model_output, tuple):
                    predictions, mode_probs = model_output
                    # Use most confident mode for autoregressive generation
                    best_mode_idx = torch.argmax(mode_probs, dim=1)
                    predictions = predictions[torch.arange(len(predictions)), best_mode_idx]
                else:
                    predictions = model_output
                    
                y = torch.cat((y, predictions[:, -1:, :]), axis=1)

            predd = y.detach().numpy()

            pred_unnormalized = dataloader.dataset.unnormalizeData(predd, org_indices)
            all_predictions.append(pred_unnormalized)

        all_np_predictions = np.concatenate(all_predictions, axis=0)[:, 1:, :2]
        self.convertAndSavePredictions(all_np_predictions)

    def convertAndSavePredictions(self, predictions):
        assert tuple(predictions.shape) == (2100, 60, 2)

        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])

        output_df.index.name = "index"
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "simple_nn_submission.csv"))


def main():
    config = {
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.005,
        "EPOCHS": 100,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.1,
        "NUM_SAMPLES": 10000,
        
        # Enhanced model parameters
        "NUM_MODES": 6,  # Multi-modal predictions
        "NUM_QUERIES": 2,
        "D_INPUT": 50 * 9,
        "D_MODEL": 256,  # Increased from 500 for better efficiency
        "N_HEAD": 8,     # Increased attention heads
        "NUM_LAYERS": 6, # Increased transformer layers
        "DROPOUT": 0.1,  # Reduced dropout
        
        # FFNN Specific parameters
        "D_EGO_FFNN_INPUT": 4,
        "FFNN_D_HIDDEN": 512,  # Reduced from 1000 for efficiency
        "FFNN_NUM_HIDDEN_LAYERS": 2,  # Reduced from 5
        "D_OUTPUT": 60 * 2 * 6,  # Multi-modal output (6 modes)
        
        # Analysis parameters (optional based on ANALYZE flag)
        "ANALYZE": True,
        "ANALYZE_NUM_BATCHES": 50,
    }

    # import os
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    trainer = AttentionAndNNTrainer(config)
    trainer.performPipeline()


if __name__ == "__main__":
    main()