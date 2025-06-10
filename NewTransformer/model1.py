from torch import nn
import torch.nn


class NewAttentionAndNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length = 110

        self.queriesEmbedding = nn.Embedding(config['NUM_QUERIES'], config['D_MODEL'])
        self.upscaleTransform = nn.Linear(config['D_INPUT'], config['D_MODEL'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["D_MODEL"],
            nhead=config["N_HEAD"],
            batch_first=True,
            dropout=config["DROPOUT"],
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["NUM_LAYERS"]
        )
        self.encoder = transformer_encoder

        # I want a FeedForwardNN here,
        ffnn_config = {
                "D_INPUT": config['NUM_QUERIES'] * config['D_MODEL'] + config['D_EGO_FFNN_INPUT'] * 50,
                "D_HIDDEN": config["FFNN_D_HIDDEN"], # New config parameter for FFNN hidden dim
                "D_OUTPUT": config["D_OUTPUT"], # This should be 2 for x,y or dx,dy
                "DROPOUT": config["DROPOUT"],
                "NUM_HIDDEN_LAYERS": config.get("FFNN_NUM_HIDDEN_LAYERS", 1)
            }
        self.prediction_nn = FeedForwardNN(ffnn_config)
        

    def forward(self, inp):
        device = self.config['DEVICE']
        num_queries = self.config['NUM_QUERIES']
        dmodel = self.config['D_MODEL']
        batch_size = inp.size()[0]
        # Expect input to be of shape (batch_size, 50, 110, 9)
        
        queries = torch.reshape(self.queriesEmbedding(torch.arange(num_queries).to(device)), (1, num_queries, dmodel)).expand((batch_size, -1, -1))
        other_agent_inp = self.upscaleTransform(torch.flatten(inp.transpose(1, 2), start_dim=2))

        precompact_inp = torch.cat((queries, other_agent_inp), axis=1)

        compact_inp = torch.flatten(self.encoder(precompact_inp)[:, :num_queries], start_dim=1)

        # Only need xpos, ypos, xvel and yvel from ego agent
        ego_inp = torch.flatten(inp[:, 0, :, :4], start_dim=1)

        full_pred_inp = torch.cat((ego_inp, compact_inp), axis=1)

        op = self.prediction_nn(full_pred_inp)

        final_op = torch.reshape(op, (-1, 60, 2))

        return final_op
    # def forward(self, inp):
    #     device = self.config['DEVICE']
    #     num_queries = self.config['NUM_QUERIES']
    #     dmodel = self.config['D_MODEL']
    #     batch_size = inp.size(0)

    #     # Input shape: (B, 50 agents, 110 time steps, 6 features)
    #     # Use only the first 50 time steps for input
    #     inp = inp[:, :, :50, :]  # shape: (B, 50, 50, 6)

    #     # Flatten time and features for each agent: (B, 50, 50 × 6 = 300)
    #     inp_flat = inp.reshape(batch_size, 50, 50 * 6)

    #     # Project agent representations to D_MODEL: (B, 50, D_MODEL)
    #     other_agent_inp = self.upscaleTransform(inp_flat)

    #     # Create query embeddings for the transformer: (B, num_queries, D_MODEL)
    #     queries = self.queriesEmbedding(torch.arange(num_queries).to(device))  # (num_queries, D_MODEL)
    #     queries = queries.unsqueeze(0).expand(batch_size, -1, -1)

    #     # Concatenate queries and agent embeddings: (B, num_queries + 50, D_MODEL)
    #     precompact_inp = torch.cat((queries, other_agent_inp), dim=1)

    #     # Pass through transformer encoder: (B, num_queries + 50, D_MODEL)
    #     transformer_out = self.encoder(precompact_inp)

    #     # Extract query outputs: (B, num_queries, D_MODEL) → flatten to (B, num_queries * D_MODEL)
    #     compact_inp = transformer_out[:, :num_queries, :].reshape(batch_size, -1)

    #     # Extract ego agent features: x, y, vx, vy from time steps → (B, 50, 4)
    #     ego_inp = inp[:, 0, :, :4].reshape(batch_size, -1)  # (B, 50 * 4)

    #     # Concatenate ego features and transformer output: (B, 50*4 + num_queries*D_MODEL)
    #     full_pred_inp = torch.cat((ego_inp, compact_inp), dim=1)

    #     # Final prediction: (B, 60*2)
    #     op = self.prediction_nn(full_pred_inp)

    #     # Reshape to (B, 60, 2)
    #     final_op = op.reshape(batch_size, 60, 2)

    #     return final_op


        


class FeedForwardNN(nn.Module):
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

