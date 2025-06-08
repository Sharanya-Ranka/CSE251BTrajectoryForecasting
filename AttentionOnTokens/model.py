from torch import nn
import torch.nn


class AttentionOnTokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_learnable_queries = config['NUM_QUERIES']
        self.embed_dim = config['D_MODEL']
        self.num_heads = config['N_HEAD']
        self.dropout = config['DROPOUT']

        self.timestepsEmbedding = nn.Embedding(15, 4)
        self.agentsEmbedding = nn.Embedding(50, 6)
        self.register_buffer('timesteps_inds', torch.arange(15,dtype=torch.long))
        self.register_buffer('agents_inds', torch.arange(50,dtype=torch.long))

        # Define your learnable queries
        # These will be the Query (Q) vectors for the attention mechanism
        self.learnable_queries = nn.Parameter(torch.randn(self.num_learnable_queries, self.embed_dim))
        self.upscaleTransform = nn.Linear(16, config['D_MODEL'])

        # MultiheadAttention module
        # q_dim, k_dim, v_dim can be different if your queries, keys, values
        # have different feature dimensions, but usually they are the same (embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True # Assuming your input tokens will be (batch, seq_len, embed_dim)
        )

        # Optional: A layer norm or feed-forward network after attention
        self.norm = nn.LayerNorm(self.embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )

        ffnn_config = {
                "D_INPUT": config['NUM_QUERIES'] * config['D_MODEL'] + config['D_EGO_FFNN_INPUT'] * 15,
                "D_HIDDEN": config["FFNN_D_HIDDEN"], # New config parameter for FFNN hidden dim
                "D_OUTPUT": config["D_OUTPUT"], # This should be 2 for x,y or dx,dy
                "DROPOUT": config["DROPOUT"],
                "NUM_HIDDEN_LAYERS": config.get("FFNN_NUM_HIDDEN_LAYERS", 1)
            }
        self.prediction_nn = FeedForwardNN(ffnn_config)

    def expandInput(self, inp):
        timesteps = 15 #len(self.timesteps_inds)
        agents = 50 #len(self.agents_inds)
        batch_size = inp.shape[0]
        
        timesteps_attrs = torch.reshape(self.timestepsEmbedding(self.timesteps_inds), (1, 1, timesteps, 4)).expand((batch_size, agents, timesteps, 4))
        agents_attrs = torch.reshape(self.agentsEmbedding(self.agents_inds), (1, agents, 1, 6)).expand((batch_size, agents, timesteps, 6))

        full_inp = torch.flatten(torch.cat((inp, timesteps_attrs, agents_attrs), dim=-1), start_dim=1, end_dim=2)
        # breakpoint()
        return full_inp
        

    def forward(self, inp):
        # input_tokens shape: (batch_size, seq_len, embed_dim)
        batch_size = inp.shape[0]

        input_tokens = self.expandInput(inp)

        # Expand learnable_queries to match batch_size for broadcasting
        # query_tensor shape: (batch_size, num_learnable_queries, embed_dim)
        query_tensor = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
        input_tokens_tensor = self.upscaleTransform(input_tokens)

        # breakpoint()

        # Perform cross-attention
        # query: your learnable queries
        # key: your input_tokens
        # value: your input_tokens
        attn_output, attn_weights = self.attention(
            query=query_tensor,
            key=input_tokens_tensor,
            value=input_tokens_tensor
        )
        # attn_output shape: (batch_size, num_learnable_queries, embed_dim)
        # attn_weights shape: (batch_size, num_learnable_queries, seq_len)
        #   (each row for a query shows how much it attended to each input token)

        # Optional: Apply feed-forward and normalization
        output = torch.flatten(self.norm(attn_output + self.feed_forward(attn_output)), start_dim=1)

        # breakpoint()
        ego_inp = torch.flatten(inp[:, 0, :, :], start_dim=1)

        full_pred_inp = torch.cat((ego_inp, output), axis=1)

        op = self.prediction_nn(full_pred_inp)

        final_op = torch.reshape(op, (-1, 60, 2))

        return final_op
        

class AttentionAndNN(nn.Module):
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
                "D_INPUT": config['NUM_QUERIES'] * config['D_MODEL'] + config['D_EGO_FFNN_INPUT'] * 15,
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
        acc = inp[:, 0, :, 2:4] - torch.roll(inp[:, 0, :,  2:4], shifts=1, dims=1) 
        ffnn_addn_inp = torch.cat((inp[:, 0, :, :4], acc), axis=-1)
        ego_inp = torch.flatten(ffnn_addn_inp, start_dim=1)

        full_pred_inp = torch.cat((ego_inp, compact_inp), axis=1)

        op = self.prediction_nn(full_pred_inp)

        final_op = torch.reshape(op, (-1, 60, 2))

        return final_op
        


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

