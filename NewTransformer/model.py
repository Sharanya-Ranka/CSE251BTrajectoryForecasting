import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EgoAgentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.input_size = 5         # Features per time step
        self.d_model = config.get("D_MODEL", 128)  # Transformer dimension
        self.nhead = config.get("NHEAD", 8)        # Number of attention heads
        self.num_encoder_layers = config.get("NUM_ENCODER_LAYERS", 6)
        self.num_decoder_layers = config.get("NUM_DECODER_LAYERS", 6)
        self.dim_feedforward = config.get("DIM_FEEDFORWARD", 512)
        self.dropout = config.get("DROPOUT", 0.1)
        self.input_steps = 50       # Historical steps
        self.output_steps = 60      # Future steps to predict
        self.output_size = 5        # [pos_diff_x, pos_diff_y, vel_x, vel_y, heading]
        
        # Input projection
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=200)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.output_size)
        
        # Learned query embeddings for future predictions
        self.future_queries = nn.Parameter(torch.randn(self.output_steps, self.d_model))
        
    def forward(self, x):
        # x: (batch_size, 50, 5)
        batch_size = x.size(0)
        
        # Project input features to model dimension
        src = self.input_projection(x)  # (batch_size, 50, d_model)
        src = src.transpose(0, 1)       # (50, batch_size, d_model)
        src = self.pos_encoder(src)     # Add positional encoding
        src = src.transpose(0, 1)       # (batch_size, 50, d_model)
        
        # Create target queries for future steps
        tgt = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 60, d_model)
        
        # Apply transformer
        output = self.transformer(src, tgt)  # (batch_size, 60, d_model)
        
        # Project to output features
        output = self.output_projection(output)  # (batch_size, 60, 5)
        
        return output

# class EgoAgentTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.seq_length = 110

#         self.queriesEmbedding = nn.Embedding(config['NUM_QUERIES'], config['D_MODEL'])
#         self.upscaleTransform = nn.Linear(config['D_INPUT'], config['D_MODEL'])
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=config["D_MODEL"],
#             nhead=config["N_HEAD"],
#             batch_first=True,
#             dropout=config["DROPOUT"],
#         )
#         transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=config["NUM_LAYERS"]
#         )
#         self.encoder = transformer_encoder

#         # I want a FeedForwardNN here,
#         ffnn_config = {
#                 "D_INPUT": config['NUM_QUERIES'] * config['D_MODEL'] + config['D_EGO_FFNN_INPUT'] * 50,
#                 "D_HIDDEN": config["FFNN_D_HIDDEN"], # New config parameter for FFNN hidden dim
#                 "D_OUTPUT": config["D_OUTPUT"], # This should be 2 for x,y or dx,dy
#                 "DROPOUT": config["DROPOUT"],
#                 "NUM_HIDDEN_LAYERS": config.get("FFNN_NUM_HIDDEN_LAYERS", 1)
#             }
#         self.prediction_nn = FeedForwardNN(ffnn_config)
        

#     def forward(self, inp):
#         device = self.config['DEVICE']
#         num_queries = self.config['NUM_QUERIES']
#         dmodel = self.config['D_MODEL']
#         batch_size = inp.size()[0]
#         # Expect input to be of shape (batch_size, 50, 110, 9)
        
#         queries = torch.reshape(self.queriesEmbedding(torch.arange(num_queries).to(device)), (1, num_queries, dmodel)).expand((batch_size, -1, -1))
#         other_agent_inp = self.upscaleTransform(torch.flatten(inp.transpose(1, 2), start_dim=2))

#         precompact_inp = torch.cat((queries, other_agent_inp), axis=1)

#         compact_inp = torch.flatten(self.encoder(precompact_inp)[:, :num_queries], start_dim=1)

#         # Only need xpos, ypos, xvel and yvel from ego agent
#         ego_inp = torch.flatten(inp[:, 0, :, :4], start_dim=1)

#         full_pred_inp = torch.cat((ego_inp, compact_inp), axis=1)

#         op = self.prediction_nn(full_pred_inp)

#         final_op = torch.reshape(op, (-1, 60, 2))

#         return final_op
        


# class FeedForwardNN(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         # Input projection layer (from D_INPUT to D_HIDDEN)
#         self.input_proj = nn.Sequential(
#             nn.Dropout(config["DROPOUT"]),
#             nn.Linear(config["D_INPUT"], config["D_HIDDEN"]),
#             nn.BatchNorm1d(config["D_HIDDEN"]),
#             nn.ReLU()
#         )

#         # Residual blocks
#         self.hidden_layers = nn.ModuleList()
#         num_hidden_layers = config.get("NUM_HIDDEN_LAYERS", 1)
#         for _ in range(num_hidden_layers):
#             self.hidden_layers.append(
#                 ResidualBlock(
#                     config["D_HIDDEN"],
#                     config["DROPOUT"]
#                 )
#             )

#         # Output layer
#         self.output_layer = nn.Sequential(
#             nn.Dropout(config["DROPOUT"]),
#             nn.Linear(config["D_HIDDEN"], config["D_OUTPUT"])
#         )

#     def forward(self, x):
#         # Input projection
#         x = self.input_proj(x)

#         # Pass through residual blocks
#         for block in self.hidden_layers:
#             x = block(x)

#         # Output layer
#         x = self.output_layer(x)
#         return x

# class ResidualBlock(nn.Module):
#     def __init__(self, dim, dropout_rate):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate), # Often a second dropout before the skip connection add
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim)
#             # No ReLU here, as it's applied *after* the addition
#         )
#         self.relu = nn.ReLU() # ReLU after the skip connection

#     def forward(self, x):
#         # Apply the block, then add the original input
#         return self.relu(x + self.block(x))




# You can choose the model here
EgoAgentNN = EgoAgentTransformer



