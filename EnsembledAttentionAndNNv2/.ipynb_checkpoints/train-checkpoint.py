import AttentionAndNN.data as dataset
import AttentionAndNN.config as config
from torch.utils.data import DataLoader
import AttentionAndNN.model as model
from AttentionAndNN.data_create import POS_DIFF_MULTIPLIER
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import utilities as utils
from example_indices import sorted_indices


class AttentionAndNNTrainer:
    def __init__(self, config):
        self.config = config

    def performPipeline(self):
        self.setUpData()
        
        self.setUpModels()
        
        self.setUpOptimizers()
        
        self.trainRouter()
        self.trainExperts()
        
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
            "NUM_REGIMES": len(config['REGIMES'] + 1)
        }
        test_config = {"INDICES": test_indices, 
                       "DATA_FILENAME": "train.npz", 
                       "NUM_REGIMES": len(config['REGIMES'] + 1)
          }

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
            "NUM_REGIMES": len(config['REGIMES'] + 1)
            "INFERENCE": True,
        }

        self.predict_dataset = dataset.AllAgentsNormalizedDataset(predict_config)

        self.predict_dataloader = DataLoader(
            self.predict_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

    def setUpModels(self):
        self.router_model = model.AttentionAndNN(self.config)

        self.experts = []
        for regime_num in range(self.config['NUM_REGIMES']+1):
            expert = model.AttentionAndNN(self.config)
            self.experts.append(expert)

    def setUpOptimizer(self):
        # COnfiguring router optimizer
        self.router_optimizer = torch.optim.Adam(
            self.router_model.parameters(), lr=self.config["ROUTER_LR"]
        )
        self.router_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.router_optimizer, patience=self.config['ROUTER_PATIENCE'], factor=0.1
        )

        # Configuring optimizers for the experts
        self.expert_optimizers = [torch.optim.Adam(
            model.parameters(), lr=self.config["EXPERT_LR"][i]
        ) for i, model in enumerate(self.experts)]
        self.expert_lr_scheduler = [ torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.config['EXPERT_PATIENCE'][i], factor=0.1
        ) for i, opt in enumerate(self.expert_optimizers) ]

    def trainRouter(self):
        # Training the router (regime classifier)
        print(f"Training Router")
        for epoch in range(self.config["ROUTER_EPOCHS"]):
            # print(f"Epoch {epoch}")
            
            self.epoch = epoch
            train_loss = self.trainRouterEpoch()
            eval_loss = self.EvalRouterEpoch()
            
            print(
                f"Router Epoch {epoch}: train_loss:{train_loss:.5f}, eval_loss:{eval_loss:.5f}"
            )

            self.scheduler.step(eval_loss)
            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch+1}, Current Learning Rate: {self.scheduler.get_last_lr()[0]}"
                )

    def trainExperts(self):
        # Running evaluation on the experts
        
        
        
        


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

            # if train_loss < 9.55:
            #     self.predict()

        return training_progress

    def computeLoss(self, true, prediction):
        # print(
        #     f"computeLoss: prediction_shape={prediction.shape}, true_shape={true.shape}"
        # )
        # breakpoint()
        loss = (1/POS_DIFF_MULTIPLIER) ** 2 * torch.mean((prediction - true) ** 2)  
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
    def train_epoch(self):
        num_batches = len(self.train_dataloader)

        self.model.train()

        train_loss, total = 0, 0
        for batch, (X, Y, indices) in enumerate(self.train_dataloader):
            # print(f"On batch={batch}")
            X = X.to(self.config["DEVICE"])
            Y = Y.to(self.config["DEVICE"])

            # breakpoint()
            
            prediction = self.model(X)
            
            loss = self.computeLoss(Y, prediction)
            
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

                prediction = self.model(X)
                
                loss = self.computeLoss(Y, prediction)

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

    def convertAndSavePredictions(self, predictions):
        assert tuple(predictions.shape) == (2100, 60, 2)

        pred_output = predictions.reshape(-1, 2)
        output_df = pd.DataFrame(pred_output, columns=["x", "y"])

        output_df.index.name = "index"
        output_df.to_csv(os.path.join(utils.SUBMISSION_DIR, "attention_and_nn.csv"))


def main():
    

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
