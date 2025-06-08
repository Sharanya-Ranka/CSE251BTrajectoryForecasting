from torch import nn
import torch.nn

class EgoAgentNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input projection layer (from D_INPUT to D_HIDDEN)
        self.input_proj = nn.Sequential(
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(config["D_INPUT"], config["D_HIDDEN"]),
            nn.BatchNorm1d(config["D_HIDDEN"]),
            nn.ReLU()
        )

        # Residual blocks
        self.hidden_layers = nn.ModuleList()
        num_hidden_layers = config.get("NUM_HIDDEN_LAYERS", 1)
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                ResidualBlock(
                    config["D_HIDDEN"],
                    config["DROPOUT"]
                )
            )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(config["DROPOUT"]),
            nn.Linear(config["D_HIDDEN"], config["D_OUTPUT"])
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.hidden_layers:
            x = block(x)

        # Output layer
        x = self.output_layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Often a second dropout before the skip connection add
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
            # No ReLU here, as it's applied *after* the addition
        )
        self.relu = nn.ReLU() # ReLU after the skip connection

    def forward(self, x):
        # Apply the block, then add the original input
        return self.relu(x + self.block(x))


# class EgoAgentNN(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         layers = []
#         # Input layer
#         layers.append(nn.Dropout(config["DROPOUT"]))
#         layers.append(nn.Linear(config["D_INPUT"], config["D_HIDDEN"]))
#         layers.append(nn.BatchNorm1d(config["D_HIDDEN"]))  # Add BatchNorm
#         layers.append(nn.ReLU())

#         # Variable number of hidden layers
#         for _ in range(
#             config.get("NUM_HIDDEN_LAYERS", 1)
#         ):  # Default to 1 hidden layer if not specified
#             layers.append(nn.Dropout(config["DROPOUT"]))
#             layers.append(nn.Linear(config["D_HIDDEN"], config["D_HIDDEN"]))
#             layers.append(nn.BatchNorm1d(config["D_HIDDEN"]))  # Add BatchNorm
#             layers.append(nn.ReLU())

#         # Output layer
#         layers.append(nn.Dropout(config["DROPOUT"]))
#         layers.append(nn.Linear(config["D_HIDDEN"], config["D_OUTPUT"]))
#         # No BatchNorm or ReLU on the output layer

#         self.network = nn.Sequential(*layers)

    # def forward(self, inp):
    #     return self.network(inp)
