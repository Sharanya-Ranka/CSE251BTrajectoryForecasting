import SimpleTransformer.data as dataset
from torch.utils.data import DataLoader
import SimpleTransformer.model as model
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import utilities as utils


class SimpleTransformerTrainer:
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
            "DATA_FILENAME": "train_data_norm.npz",
        }
        test_config = {"INDICES": test_indices, "DATA_FILENAME": "train_data_norm.npz"}

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
            "DATA_FILENAME": "test_data_norm.npz",
            "INFERENCE": True,
        }

        self.predict_dataset = dataset.EgoAgentNormalizedDataset(predict_config)

        self.predict_dataloader = DataLoader(
            self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

    def setUpModel(self):
        self.model = model.EgoAgentTransformer(self.config)
        print(self.model)

    def setUpOptimizer(self):
        # Define the optimizer being used
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["LEARNING_RATE"]
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.3)

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
            self.scheduler.step()

            if self.config.get("ANALYZE"):
                unnormalized_score = self.analyze(self.train_dataloader)
                print(f"Unnormalized score (train) {unnormalized_score}")

                unnormalized_score = self.analyze(self.test_dataloader)
                print(f"Unnormalized score (test) {unnormalized_score}")

                # if unnormalized_score < best_unnormalized_score:
                #     best_unnormalized_score = unnormalized_score
                #     self.predict()

        return training_progress

    def computeLoss(self, true, prediction):
        # print(
        #     f"computeLoss: prediction_shape={prediction.shape}, true_shape={true.shape}"
        # )
        # breakpoint()
        loss = torch.mul(
            400, torch.mean((true[:, :, :2] - prediction[:, :, :2]) ** 2)
        )  # + torch.mul(5, torch.mean((true[:, :, 2:5] - prediction[:, :, 2:5]) ** 2))
        return loss

    # Training function
    def train_epoch(self):
        num_batches = len(self.train_dataloader)

        self.model.train()

        train_loss, total = 0, 0
        for batch, (X, y, _) in enumerate(self.train_dataloader):
            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])

            # Compute prediction error
            prediction = self.model(X, y)
            loss = self.computeLoss(y, prediction)
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

                # Compute prediction
                # Assuming the model forward pass is the same as during training
                # (takes X and y as input, likely for teacher forcing in decoder)
                prediction = self.model(X, torch.zeros_like(y))

                # Compute loss
                loss = self.computeLoss(y, prediction)
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
            predip = y[:, :1, :]
            for i in range(inference_steps):
                predop = self.model(X, predip)
                predip = torch.cat((predip, predop[:, -1:, :]), axis=1)

            # breakpoint()
            yd = y.detach().numpy()
            predd = predip.detach().numpy()

            true_unnormalized = dataloader.dataset.unnormalizeData(yd, org_indices)
            pred_unnormalized = dataloader.dataset.unnormalizeData(predd, org_indices)
            unnormalized_metric = dataloader.dataset.computeUnnormalizedMetric(
                true_unnormalized, pred_unnormalized
            )
            total_unnormalized += unnormalized_metric

        return total_unnormalized / min(num_batches, len(dataloader))

    def predict(self):
        all_predictions = []
        dataloader = self.predict_dataloader
        inference_steps = 60

        self.model.eval()

        total_unnormalized = 0
        for batch, (X, y, org_indices) in enumerate(dataloader):

            X, y = X.to(self.config["DEVICE"]), y.to(self.config["DEVICE"])
            for i in range(inference_steps):
                predictions = self.model(X, y)
                y = torch.cat((y, predictions[:, -1:, :]), axis=1)

            predd = y.detach().numpy()

            pred_unnormalized = dataloader.dataset.unnormalizeData(predd, org_indices)
            # print(f"Pred unnorm shape={pred_unnormalized.shape}")
            # print(f"Pred={pred_unnormalized}")
            # breakpoint()
            all_predictions.append(pred_unnormalized)
            # predictions.append(pred_unnormalized)

        all_np_predictions = np.concatenate(all_predictions, axis=0)[:, 1:, :2]
        self.convertAndSavePredictions(all_np_predictions)

    def convertAndSavePredictions(self, predictions):
        assert tuple(predictions.shape) == (2100, 60, 2)

        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])

        output_df.index.name = "index"
        os.makedirs(utils.SUBMISSION_DIR, exist_ok=True)
        output_df.to_csv(
            os.path.join(utils.SUBMISSION_DIR, "simple_transformer_submission.csv")
        )


def main():
    config = {
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 20,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.3,
        "NUM_SAMPLES": 5000,
        # Transformer specific parameters
        "D_INPUT": 6,
        "D_OUTPUT": 6,
        "D_MODEL": 24,  # Example value for the model dimension
        "NHEAD": 4,  # Example value for the number of attention heads
        "NUM_ENCODER_LAYERS": 6,  # Example value for the number of encoder layers
        "NUM_DECODER_LAYERS": 6,  # Example value for the number of decoder layers
        "DIM_FEEDFORWARD": 50,  # Example value for the feedforward dimension (often 4 * D_MODEL)
        "DROPOUT": 0.1,  # Example dropout rate
        # Analysis parameters (optional based on ANALYZE flag)
        "ANALYZE": True,
        "ANALYZE_NUM_EXAMPLES": 100,
    }

    trainer = SimpleTransformerTrainer(config)
    trainer.performPipeline()

    # config1 = {
    #     "BATCH_SIZE": 32,
    #     "LEARNING_RATE": 0.0005,
    #     "EPOCHS": 20,
    #     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    #     "TEST_SIZE": 0.3,
    #     "NUM_SAMPLES": 5000,
    #     # Transformer specific parameters
    #     "D_INPUT": 6,
    #     "D_OUTPUT": 6,
    #     "D_MODEL": 24,  # Example value for the model dimension
    #     "NHEAD": 4,  # Example value for the number of attention heads
    #     "NUM_ENCODER_LAYERS": 6,  # Example value for the number of encoder layers
    #     "NUM_DECODER_LAYERS": 6,  # Example value for the number of decoder layers
    #     "DIM_FEEDFORWARD": 50,  # Example value for the feedforward dimension (often 4 * D_MODEL)
    #     "DROPOUT": 0.1,  # Example dropout rate
    #     # Analysis parameters (optional based on ANALYZE flag)
    #     "ANALYZE": True,
    #     "ANALYZE_NUM_EXAMPLES": 100,
    # }

    # config = {
    #     "BATCH_SIZE": 16,
    #     "LEARNING_RATE": 0.0005,
    #     "EPOCHS": 20,
    #     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    #     "TEST_SIZE": 0.1,
    #     "NUM_SAMPLES": 1000,
    #     # Transformer specific parameters
    #     "D_MODEL": 6,  # Example value for the model dimension
    #     "NHEAD": 2,  # Example value for the number of attention heads
    #     "NUM_ENCODER_LAYERS": 3,  # Example value for the number of encoder layers
    #     "NUM_DECODER_LAYERS": 3,  # Example value for the number of decoder layers
    #     "DIM_FEEDFORWARD": 30,  # Example value for the feedforward dimension (often 4 * D_MODEL)
    #     "DROPOUT": 0.3,  # Example dropout rate
    #     # Analysis parameters (optional based on ANALYZE flag)
    #     "ANALYZE": True,
    #     "ANALYZE_NUM_EXAMPLES": 100,
    # }
