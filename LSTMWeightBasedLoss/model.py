from torch import nn
import torch.nn

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