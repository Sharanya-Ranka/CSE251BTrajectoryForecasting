import AttentionAndNN.data as dataset
from AttentionAndNN.data_create import POS_DIFF_MULTIPLIER
from torch.utils.data import DataLoader
import AttentionAndNN.model as model
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import utilities as utils

SECOND_MODEL_MULTIPLIER = 5

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
        self.model1 = model.AttentionAndNN(self.config)
        self.model1.to(self.config["DEVICE"])
        print(self.model1)

        self.model2 = model.AttentionAndNN(self.config)
        self.model2.to(self.config["DEVICE"])
        print(self.model2)

    def setUpOptimizer(self):
        # Define the optimizer being used
        self.optimizer1 = torch.optim.Adam(
            self.model1.parameters(), lr=self.config["LEARNING_RATE"]
        )

        self.scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer1, patience=5, factor=0.1
        )

        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(), lr=self.config["LEARNING_RATE"]
        )

        self.scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer2, patience=5, factor=0.1
        )

    def train(self):
        training_progress = {}
        print(f"Training model 1")
        for epoch in range(self.config["EPOCHS"]):
                print(f"Performing epoch {epoch}")
                self.epoch = epoch
                train_loss = self.train_epoch1()
                eval_loss = self.eval_epoch1()
                print(
                    f"Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
                )
                training_progress[epoch] = {
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                }
    
                self.scheduler1.step(eval_loss)
                print(
                    f"Epoch {epoch+1}, Current Learning Rate: {self.scheduler1.get_last_lr()[0]}"
                )

        self.model1.eval()

        print(f"Training model 2")
        for epoch in range(self.config["EPOCHS"]):
                print(f"Performing epoch {epoch}")
                self.epoch = epoch
                train_loss = self.train_epoch2()
                eval_loss = self.eval_epoch2()
                print(
                    f"Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
                )
                training_progress[epoch] = {
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                }
    
                self.scheduler2.step(eval_loss)
                print(
                    f"Epoch {epoch+1}, Current Learning Rate: {self.scheduler2.get_last_lr()[0]}"
                )
            
        # for i, model in enumerate([self.model1, self.model2]):
        #     cur_scheduler = [self.scheduler1, self.scheduler2][i]
        #     print(f"Model:{i}")
        #     for epoch in range(self.config["EPOCHS"]):
        #         print(f"Performing epoch {epoch}")
        #         self.epoch = epoch
        #         train_loss = self.train_epoch()
        #         eval_loss = self.eval_epoch()
        #         print(
        #             f"Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
        #         )
        #         training_progress[epoch] = {
        #             "train_loss": train_loss,
        #             "eval_loss": eval_loss,
        #         }
    
        #         cur_scheduler.step(eval_loss)
        #         print(
        #             f"Epoch {epoch+1}, Current Learning Rate: {cur_scheduler.get_last_lr()[0]}"
        #         )
    
        if self.config.get("ANALYZE"):
            origspace_score_tr = self.analyze(self.train_dataloader)
            print(f"OrigSpace score (train) {origspace_score_tr}")
    
                    # origspace_score_te = self.analyze(self.test_dataloader)
                    # print(f"OrigSpace score (test) {origspace_score_te}")
    
                    # if unnormalized_score < best_unnormalized_score:
                    #     best_unnormalized_score = unnormalized_score
                    #     self.predict()

        return training_progress

    def computeLoss(self, true, prediction):
        # print(
        #     f"computeLoss: prediction_shape={prediction.shape}, true_shape={true.shape}"
        # )
        # breakpoint()
        loss = (1 / POS_DIFF_MULTIPLIER)**2 * torch.mean((true - prediction) ** 2)  
                # + 0 * torch.mean((true[:, 0, :, [4, 8]] - - prediction[:, 0, :, [4, 8]]) ** 2)  \
                # + 0 * torch.mean((true - prediction) ** 2)

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

    # def analyzeSpecific(self, true, prediction, indices):
    #     losses = torch.mul(
    #         100, torch.mean((true[:, 0, :, 2:4] - prediction[:, 0, :, 2:4]) ** 2, dim=2)
    #     )
    #     breakpoint()

    #     self.viewPercentilesWithSampleIndices(losses.cpu().detach().numpy(), indices)

    # Training function
    def train_epoch1(self):
        num_batches = len(self.train_dataloader)

        self.model1.train()

        train_loss, total = 0, 0
        for batch, (X, Y, indices) in enumerate(self.train_dataloader):
            # print(f"On batch={batch}")
            X = X.to(self.config["DEVICE"])
            Y = Y.to(self.config["DEVICE"])
            
            prediction = self.model1(X)
            
            loss = self.computeLoss(Y, prediction)
            
            train_loss += loss.item()

            # Backpropagation
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()

        average_train_loss = float(train_loss / num_batches)
        return average_train_loss

    # Training function
    def train_epoch2(self):
        num_batches = len(self.train_dataloader)

        self.model2.train()

        train_loss, total = 0, 0
        for batch, (X, Y, indices) in enumerate(self.train_dataloader):
            # print(f"On batch={batch}")
            X = X.to(self.config["DEVICE"])
            Y = Y.to(self.config["DEVICE"])
            
            prediction = self.model2(X)
            
            loss = self.computeLoss((Y - self.model1(X)) * SECOND_MODEL_MULTIPLIER, prediction)
            
            train_loss += loss.item()

            # Backpropagation
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()

        average_train_loss = float(train_loss / num_batches)
        return average_train_loss

    def eval_epoch1(self):
        # Set the model to evaluation mode.
        # This disables dropout and uses accumulated batch norm statistics (if any).
        self.model1.eval()

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

                prediction = self.model1(X)
                
                loss = self.computeLoss(Y, prediction)

                # breakpoint()
                eval_loss += loss.item()

        # Calculate average loss
        average_eval_loss = float(eval_loss / num_batches)

        return average_eval_loss

    def eval_epoch2(self):
        # Set the model to evaluation mode.
        # This disables dropout and uses accumulated batch norm statistics (if any).
        self.model2.eval()

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

                prediction = self.model2(X)
                
                loss = self.computeLoss((Y - self.model1(X)) * SECOND_MODEL_MULTIPLIER, prediction)

                # breakpoint()
                eval_loss += loss.item()

        # Calculate average loss
        average_eval_loss = float(eval_loss / num_batches)

        return average_eval_loss

    def analyze(self, dataloader: DataLoader):
        num_batches = self.config["ANALYZE_NUM_BATCHES"]

        self.model1.eval()
        self.model2.eval()
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

                prediction1 = self.model1(X)
                prediction2 = self.model2(X) / SECOND_MODEL_MULTIPLIER
                prediction = (prediction1 + prediction2)

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
                # if self.epoch > 10:
                #     breakpoint()
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
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "simple_nn_submission.csv"))


def main():
    config = {
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.005,
        "EPOCHS": 20,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "TEST_SIZE": 0.1,
        "NUM_SAMPLES": 10000,
        # Transformer specific parameters
        "NUM_QUERIES" : 2,
        # "D_INPUT": 50 * 9 + 50 * 6 + 7,
        "D_INPUT": 50 * 9,
        "D_MODEL": 500,
        "N_HEAD": 2,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.1,  # Example dropout rate
        # FFNN Specific parameters
        "D_EGO_FFNN_INPUT" : 6,
        "FFNN_D_HIDDEN" : 1000,
        "FFNN_NUM_HIDDEN_LAYERS":3,
        "D_OUTPUT": 60 * 2,
        
        # Analysis parameters (optional based on ANALYZE flag)
        "ANALYZE": True,
        "ANALYZE_NUM_BATCHES": 50,
    }

    # import os
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    trainer = AttentionAndNNTrainer(config)
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
