from torch import nn
import torch.nn


class PytorchEampleTransformer(nn.Transformer):
    def __init__(
        self,
        ntoken,
        d_model,
        nhead,
        dim_feedforward,
        num_encoder_layers,
        num_decoder_layers,
        dropout=0.5,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


class EgoAgentTransformer2(nn.Transformer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_encoder_layers,
        num_decoder_layers,
        dropout=0.5,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.model_type = "Transformer"
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def forward(self, src, tgt, has_mask=True):
        # Pass source and target into the transformer (Check masks!)
        tgt_pred = super().forward(src, tgt)

        # Return the output vector (tgt_pred) (loss calculated outside)
        return tgt_pred


class EgoAgentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lin1 = nn.Linear(config["D_INPUT"], config["D_MODEL"], dtype=torch.float64)
        self.dp1 = nn.Dropout(p=config["DROPOUT"])
        self.transformer = nn.Transformer(
            d_model=config["D_MODEL"],
            nhead=config["NHEAD"],
            num_encoder_layers=config["NUM_ENCODER_LAYERS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],  # Initialized from config
            dropout=config["DROPOUT"],  # Initialized from config
            device=config["DEVICE"],  # Initialized from config (already identified)
            batch_first=True,
            dtype=torch.float64,
        )
        self.lin2 = nn.Linear(
            config["D_MODEL"], config["D_OUTPUT"], dtype=torch.float64
        )

    def forward(self, src, tgt):
        # breakpoint()
        # Transform input dimensions to model dimensions
        src_t = self.lin1(src)
        src_t = self.dp1(src_t)

        tgt_t = self.lin1(tgt)
        tgt_t = self.dp1(tgt_t)

        # Pass source and target into the transformer (Check masks!)
        tgt_pred_t = self.transformer(src_t, tgt_t)

        tgt_pred = self.lin2(tgt_pred_t)

        # Return the output vector (tgt_pred) (loss calculated outside)
        return tgt_pred

    def forward1(self, src, tgt):

        # Pass source and target into the transformer (Check masks!)
        tgt_pred = self.transformer(src, tgt)
        # Return the output vector (tgt_pred) (loss calculated outside)
        return tgt_pred
