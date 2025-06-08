import EnsembledAttentionAndNNv2.data as dataset
from torch.utils.data import DataLoader
import EnsembledAttentionAndNNv2.model as model
from EnsembledAttentionAndNNv2.data_create import POS_DIFF_MULTIPLIER
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import utilities as utils
from example_indices import sorted_indices


class RouterTrainer:
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
        total_indices = np.arange(self.config["NUM_SAMPLES"])
        #[ind for ind in range(self.config['NUM_SAMPLES']) if ind in set(sorted_indices[-500:-100])] 
        print(f"Length of indices={len(total_indices)}")
        train_indices, test_indices = train_test_split(
            total_indices, test_size=self.config["TEST_SIZE"], random_state=42
        )

        train_config = {
            "INDICES": train_indices,
            "DATA_FILENAME": "train.npz",
            "REGIMES": self.config["REGIMES"],
        }
        test_config = {"INDICES": test_indices, 
                       "DATA_FILENAME": "train.npz", 
                       "REGIMES": self.config["REGIMES"],
          }

        self.train_dataset = dataset.AllAgentsNormalizedRouterDataset(train_config)
        self.test_dataset = dataset.AllAgentsNormalizedRouterDataset(test_config)

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
            "REGIMES": self.config["REGIMES"],
            "INFERENCE": True,
        }

        self.predict_dataset = dataset.AllAgentsNormalizedRouterDataset(predict_config)

        self.predict_dataloader = DataLoader(
            self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

    def setUpModel(self):
        self.model = model.EgoAgentNN(self.config)
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
            train_loss, train_acc = self.train_epoch()
            eval_loss, eval_acc = self.eval_epoch()
            print(
                f"Router Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}, train_acc:{train_acc:.5f}, eva_acc:{eval_acc:.5f}"
            )

            self.scheduler.step(eval_loss)
            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch+1}, Current Learning Rate: {self.scheduler.get_last_lr()[0]}"
                )

    def computeLoss(self, true, prediction):
        # !! TODO
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(prediction, true)
        # loss = 
        return loss

    def computeCorrectClassifications(self, true, prediction):
        pred_classes = torch.argmax(prediction, dim=-1)
        correct_mean = torch.mean(torch.eq(true, pred_classes).float())

        if np.random.random() < 0.01:
            print(f"Confusion matrix")
            print(confusion_matrix(true.cpu().detach().numpy(), pred_classes.cpu().detach().numpy()))
            
        return correct_mean
        
    def train_epoch(self):
        num_batches = len(self.train_dataloader)
        self.model.train()

        train_loss, total_correct_clss = 0, 0
        for batch, (X, Y, indices) in enumerate(self.train_dataloader):
            X = X.to(self.config["DEVICE"])
            Y = Y.to(self.config["DEVICE"])

            # breakpoint()
            
            prediction = self.model(X)
            
            loss = self.computeLoss(Y, prediction)
            mean_correct_classifications = self.computeCorrectClassifications(Y, prediction)
            
            train_loss += loss.item()
            total_correct_clss += mean_correct_classifications

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        average_train_loss = float(train_loss / num_batches)
        average_train_acc = float(total_correct_clss / num_batches)
        
        return average_train_loss, average_train_acc

    def eval_epoch(self):
        self.model.eval()

        eval_loss, total_correct_clss = 0, 0
        num_batches = len(self.test_dataloader)
        
        with torch.no_grad():
            for batch, (X, Y, _) in enumerate(self.test_dataloader):
                X = X.to(self.config["DEVICE"])
                Y = Y.to(self.config["DEVICE"])

                prediction = self.model(X)
                
                loss = self.computeLoss(Y, prediction)
                mean_correct_classifications = self.computeCorrectClassifications(Y, prediction)

                # breakpoint()
                eval_loss += loss.item()
                total_correct_clss += mean_correct_classifications

        # Calculate average loss
        average_eval_loss = float(eval_loss / num_batches)
        average_eval_acc = float(total_correct_clss / num_batches)

        return average_eval_loss, average_eval_acc

    def saveModel(self):
        torch.save(self.model.state_dict(), self.config['MODEL_SAVE_PATH'])

    def loadModel(self):
        router_model = model.EgoAgentNN(self.config)
        state_dict_path = self.config['MODEL_SAVE_PATH']
        loaded_state_dict = torch.load(state_dict_path)
        router_model.load_state_dict(loaded_state_dict)
        
        router_model.to(self.config["DEVICE"])
        print(router_model)
        
        return router_model

    def loadPredictionData(self):
        # For predictions
        predict_indices = np.arange(2100)

        predict_config = {
            "INDICES": predict_indices,
            "DATA_FILENAME": "test.npz",
            "REGIMES": self.config["REGIMES"],
            "INFERENCE": True,
        }

        predict_dataset = dataset.AllAgentsNormalizedRouterDataset(predict_config)

        predict_dataloader = DataLoader(
            predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

        return predict_dataloader

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

                prediction = self.model(X)

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
                # if self.epoch > 60:
                #     breakpoint()
                total_unnormalized += unnormalized_metric

        return total_unnormalized / min(num_batches, len(dataloader))

    def predict(self):
        all_predictions = []
        dataloader = self.predict_dataloader

        self.model.eval()

        total_unnormalized = 0
        with torch.no_grad():
            for batch, (X, _, org_indices) in enumerate(dataloader):
                X = X.to(self.config["DEVICE"])
                # Y = Y.to(self.config["DEVICE"])

                prediction = self.model(X)
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
                # Y = Y.to(self.config["DEVICE"])

                prediction = model(X)
                all_predictions.append(prediction)

        all_np_predictions = (torch.nn.Softmax(dim=1)(torch.cat(all_predictions, dim=0))).cpu().detach().numpy()

        return all_np_predictions
        

    def convertAndSavePredictions(self, predictions):
        assert tuple(predictions.shape) == (2100, 60, 2)

        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])

        output_df.index.name = "index"
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "attention_and_nn.csv"))

