from torch import nn
import torch.nn


# class EgoAgentNN(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.dp0 = nn.Dropout(config["DROPOUT"])
#         self.lin1 = nn.Linear(
#             config["D_INPUT"], config["D_HIDDEN"], dtype=torch.float64
#         )
#         self.dp1 = nn.Dropout(config["DROPOUT"])
#         self.lin2 = nn.Linear(
#             config["D_HIDDEN"], config["D_HIDDEN"], dtype=torch.float64
#         )
#         self.dp2 = nn.Dropout(config["DROPOUT"])
#         self.lin3 = nn.Linear(
#             config["D_HIDDEN"], config["D_OUTPUT"], dtype=torch.float64
#         )

#     def forward(self, inp):
#         # breakpoint()
#         l1_inp = self.dp0(inp)
#         l1_op = nn.ReLU()(self.lin1(l1_inp))
#         l2_inp = self.dp1(l1_op)
#         l2_op = nn.ReLU()(self.lin2(l2_inp))
#         l3_inp = self.dp2(l2_op)
#         l3_op = nn.ReLU()(self.lin3(l3_inp))

#         op = l3_op

#         return op


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
