import EnsembledAttentionAndNN.data as dataset
# import AttentionAndNN.config as config
from torch.utils.data import DataLoader
import EnsembledAttentionAndNN.model as model
from EnsembledAttentionAndNN.data_create import POS_DIFF_MULTIPLIER
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import utilities as utils
from example_indices import sorted_indices


class ExpertTrainer:
    def __init__(self, config):
        self.config = config

    def performPipeline(self):
        self.setUpData()
        
        self.setUpModel()
        
        self.setUpOptimizer()
        
        self.train()

        self.saveModel()
        
        # self.predict()

    def setUpData(self):
        valid_indices = set(sorted_indices[self.config['MIN_IND']:self.config['MAX_IND']])
        total_indices = [ind for ind in range(self.config['NUM_SAMPLES']) if ind in valid_indices ]
        
        # np.arange(self.config["NUM_SAMPLES"])
        #[ind for ind in range(self.config['NUM_SAMPLES']) if ind in set(sorted_indices[-500:-100])]
        
        print(f"Length of indices={len(total_indices)}")
        
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )

        train_config = {
            "INDICES": train_indices,
            "DATA_FILENAME": "train.npz",
        }
        test_config = {"INDICES": test_indices, 
                       "DATA_FILENAME": "train.npz"}

        self.train_dataset = dataset.AllAgentsNormalizedDataset(train_config)
        self.test_dataset = dataset.AllAgentsNormalizedDataset(test_config)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

        # For predictions !!TODO
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
        print(self.model)

    def setUpOptimizer(self):
        # Define the optimizer being used
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["LEARNING_RATE"]
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.config['PATIENCE'], factor=0.1
        )

    def train(self):
        for epoch in range(self.config["EPOCHS"]):
            self.epoch = epoch
            train_loss = self.train_epoch()
            eval_loss = self.eval_epoch()
            print(
                f"Expert Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
            )

            self.scheduler.step(eval_loss)
            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch+1}, Current Learning Rate: {self.scheduler.get_last_lr()[0]}"
                )

            if self.config.get("ANALYZE") and self.epoch % 5 == 0:
                origspace_score_tr = self.analyze(self.train_dataloader)
                print(f"OrigSpace score (train) {origspace_score_tr}")

    def computeLoss(self, true, prediction):
        loss = (1/POS_DIFF_MULTIPLIER) ** 2 * torch.mean((prediction - true) ** 2)
        return loss

    def train_epoch(self):
        num_batches = len(self.train_dataloader)
        self.model.train()
        train_loss = 0
        for batch, (X, Y, indices) in enumerate(self.train_dataloader):
            X = X.to(self.config["DEVICE"])
            Y = Y.to(self.config["DEVICE"])

            # breakpoint()
            
            pred = self.model(X)
            prediction = torch.reshape(pred, (-1, 60, 2))
            
            loss = self.computeLoss(Y, prediction)
            
            train_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        average_train_loss = float(train_loss / num_batches)
        return average_train_loss

    def eval_epoch(self):
        self.model.eval()
        eval_loss = 0
        num_batches = len(self.test_dataloader)
        
        with torch.no_grad():
            for batch, (X, Y, _) in enumerate(self.test_dataloader):
                X = X.to(self.config["DEVICE"])
                Y = Y.to(self.config["DEVICE"])

                pred = self.model(X)
                prediction = torch.reshape(pred, (-1, 60, 2))
                
                loss = self.computeLoss(Y, prediction)
                # breakpoint()
                eval_loss += loss.item()

        # Calculate average loss
        average_eval_loss = float(eval_loss / num_batches)
        return average_eval_loss

    def analyze(self, dataloader: DataLoader):
        num_batches = self.config["ANALYZE_NUM_BATCHES"]
        self.model.eval()
        
        total_unnormalized = 0
        with torch.no_grad():
            for batch, (X, Y, org_indices) in enumerate(dataloader):
                if batch >= num_batches:
                    break
                X = X.to(self.config["DEVICE"])
                Y = Y.to(self.config["DEVICE"])

                pred = self.model(X)
                prediction = torch.reshape(pred, (-1, 60, 2))

                true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    Y.cpu().detach().numpy(), org_indices, indicator="true"
                )
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    prediction.cpu().detach().numpy(),
                    org_indices,
                    indicator="prediction",
                )
                unnormalized_metric = dataloader.dataset.computeOriginalSpaceMetric(
                    true_unnormalized, pred_unnormalized
                )
                # if self.epoch > 60:
                #     breakpoint()
                total_unnormalized += unnormalized_metric

        return total_unnormalized / min(num_batches, len(dataloader))

    def saveModel(self):
        torch.save(self.model.state_dict(), self.config['MODEL_SAVE_PATH'])

    def loadModel(self):
        model_config = self.config
        state_dict_path = model_config['MODEL_SAVE_PATH']
        expert_model = model.AttentionAndNN(model_config)
        
        loaded_state_dict = torch.load(state_dict_path)
        expert_model.load_state_dict(loaded_state_dict)
        
        expert_model.to(model_config["DEVICE"])
        print(expert_model)
        
        return expert_model

    def loadPredictionData(self):
        # For predictions
        predict_indices = np.arange(2100)

        predict_config = {
            "INDICES": predict_indices,
            "DATA_FILENAME": "test.npz",
            "INFERENCE": True,
        }

        predict_dataset = dataset.AllAgentsNormalizedDataset(predict_config)
        

        predict_dataloader = DataLoader(
            predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

        return predict_dataloader

    def predict(self):
        all_predictions = []
        dataloader = self.predict_dataloader

        self.model.eval()

        total_unnormalized = 0
        with torch.no_grad():
            for batch, (X, _, org_indices) in enumerate(dataloader):
                X = X.to(self.config["DEVICE"])
                # Y = Y.to(self.config["DEVICE"])

                pred = self.model(X)
                prediction = torch.reshape(pred, (-1, 60, 2))

                # true_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                #     X[:, 0, 50:, :2].cpu().detach().numpy(), org_indices, indicator="true"
                # )
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    prediction.cpu().detach().numpy(),
                    org_indices,
                    indicator="prediction",
                )
                all_predictions.append(pred_unnormalized)

        all_np_predictions = np.concatenate(all_predictions, axis=0)
        breakpoint()
        self.convertAndSavePredictions(all_np_predictions)

    def predictFromOutside(self, dataloader, model):
        all_predictions = []

        model.eval()

        total_unnormalized = 0
        with torch.no_grad():
            for batch, (X, _, org_indices) in enumerate(dataloader):
                X = X.to(self.config["DEVICE"])

                pred = model(X)
                prediction = torch.reshape(pred, (-1, 60, 2))
                
                pred_unnormalized = dataloader.dataset.getOriginalSpacePredictions(
                    prediction.cpu().detach().numpy(),
                    org_indices,
                    indicator="prediction",
                )
                all_predictions.append(pred_unnormalized)

        all_np_predictions = np.concatenate(all_predictions, axis=0)

        return all_np_predictions
        

    def convertAndSavePredictions(self, predictions):
        assert tuple(predictions.shape) == (2100, 60, 2)

        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])

        output_df.index.name = "index"
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "attention_and_nn.csv"))


# def main():
#     trainer = AttentionAndNNTrainer(config)
#     trainer.performPipeline()

