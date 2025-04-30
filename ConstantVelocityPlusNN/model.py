from torch import nn
import torch.nn


class EgoAgentNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        layers = []
        # Input layer
        layers.append(nn.Dropout(config["DROPOUT"]))
        layers.append(nn.Linear(config["D_INPUT"], config["D_HIDDEN"]))
        layers.append(nn.BatchNorm1d(config["D_HIDDEN"]))  # Add BatchNorm
        layers.append(nn.ReLU())

        # Variable number of hidden layers
        for _ in range(
            config.get("NUM_HIDDEN_LAYERS", 1)
        ):  # Default to 1 hidden layer if not specified
            layers.append(nn.Dropout(config["DROPOUT"]))
            layers.append(nn.Linear(config["D_HIDDEN"], config["D_HIDDEN"]))
            layers.append(nn.BatchNorm1d(config["D_HIDDEN"]))  # Add BatchNorm
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Dropout(config["DROPOUT"]))
        layers.append(nn.Linear(config["D_HIDDEN"], config["D_OUTPUT"]))
        # No BatchNorm or ReLU on the output layer

        self.network = nn.Sequential(*layers)

    def forward(self, inp):
        return self.network(inp)
