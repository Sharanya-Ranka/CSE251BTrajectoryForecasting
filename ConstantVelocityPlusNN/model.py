from torch import nn
import torch.nn


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

#     def forward(self, inp):
#         return self.network(inp)


# class EgoAgentNN(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         d_input = config["D_INPUT"]
#         d_hidden = config["D_HIDDEN"]
#         d_output = config["D_OUTPUT"]
#         dropout = config["DROPOUT"]

#         self.input_layer = nn.Sequential(
#             nn.Linear(d_input, d_hidden),
#             nn.BatchNorm1d(d_hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         # Define 4 residual blocks (for total 5 layers including input)
#         self.res_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(d_hidden, d_hidden),
#                 nn.BatchNorm1d(d_hidden),
#                 nn.ReLU(),
#                 nn.Dropout(dropout)
#             )
#             for _ in range(4)
#         ])

#         # Output layer
#         self.output_layer = nn.Linear(d_hidden, d_output)

#         # For skip connection from input to output (only if shapes match)
#         self.enable_skip = d_input == d_output
#         if self.enable_skip:
#             self.skip_proj = nn.Identity()  # or nn.Linear(d_input, d_output) if needed
#         else:
#             self.skip_proj = None

#     def forward(self, x):
#         # Save input for skip connection
#         x_input = x

#         x = self.input_layer(x)

#         # Residual connections
#         for block in self.res_blocks:
#             x = x + block(x)

#         out = self.output_layer(x)

#         # Skip connection: add original input if allowed
#         if self.enable_skip:
#             out = out + self.skip_proj(x_input)

#         return out


class EgoAgentNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.input_size = 5         # Features per time step
        self.hidden_size = 128      # LSTM hidden units
        self.num_layers = 2         # LSTM layers
        self.output_steps = 60      # How many steps to predict
        self.output_size = 5        # [pos_diff_x, pos_diff_y, vel_x, vel_y, heading]

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config["DROPOUT"]
        )

        # Use final hidden state to predict the full future
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_steps * self.output_size)  # Flattened output
        )

    def forward(self, x):
        # x: (batch_size, 50, 5)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, 50, hidden_size)

        # Use the last time step's hidden state
        final_hidden = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)

        out = self.decoder(final_hidden)  # shape: (batch_size, 60 * 5)
        out = out.view(-1, 60, 5)         # reshape to (batch_size, 60, 5)

        return out