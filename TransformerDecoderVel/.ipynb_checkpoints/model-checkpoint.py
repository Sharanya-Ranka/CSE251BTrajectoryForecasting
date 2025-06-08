from torch import nn
import torch.nn


class DecoderOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length = 110

        self.time_embedding = nn.Embedding(self.seq_length, 7)
        self.agent_embedding = nn.Embedding(50, 6)

        self.lin_inp = nn.Linear(config["D_INPUT"], config["D_MODEL"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["D_MODEL"],
            nhead=config["N_HEAD"],
            batch_first=True,
            dropout=config["DROPOUT"],
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["NUM_LAYERS"]
        )

        self.lin_op = nn.Linear(config["D_MODEL"], config["D_OUTPUT"])

        self.network = transformer_encoder

    def transformInput(self, inp):
        timesteps = inp.shape[2]
        agents = inp.shape[1]

        agent_embeddings = torch.reshape(self.agent_embedding(torch.arange(agents).to(self.config['DEVICE'])), (1, agents, 1, -1)).expand((inp.shape[0], agents, inp.shape[2], -1))
        input_with_agents = torch.cat((inp, agent_embeddings), dim=-1)
        reshape_inp1 = input_with_agents.transpose(1, 2).flatten(start_dim=2)
        
        time_embeddings = torch.reshape(self.time_embedding(torch.arange(timesteps).to(self.config['DEVICE'])), (1, timesteps, -1)).expand((inp.shape[0], timesteps, -1))
        input_with_timesteps = torch.cat((reshape_inp1, time_embeddings), dim=-1)

        # breakpoint()

        return input_with_timesteps
        

    def forward(self, inp):
        # reshaped_inp = self.transformInput(inp)
        # breakpoint()

        reshaped_inp = inp.transpose(1, 2).flatten(start_dim=2)
        inp1 = self.lin_inp(reshaped_inp)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.seq_length
        ).to(self.config["DEVICE"])

        output = self.network(inp1, mask=causal_mask)

        op = self.lin_op(output)
        reshaped_op = op.reshape((op.shape[0], op.shape[1], 50, 9)).transpose(1, 2)
        # breakpoint()

        return reshaped_op

    def generatev2(self, inp, steps=60):
        reshaped_inp = inp.transpose(1, 2).flatten(
            start_dim=2
        )  # (examples, agents, timesteps, attrs) -> (examples, timesteps, agents * attrs)
        inp1 = self.lin_inp(reshaped_inp)
        org_steps = inp1.shape[1]

        input_cast = torch.zeros((inp1.shape[0], org_steps + steps, inp1.shape[2])).to(
            self.config["DEVICE"]
        )
        input_cast[:, :org_steps, :] = inp1

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.seq_length
        ).to(self.config["DEVICE"])

        for step in range(steps):
            step_output = self.network(input_cast, mask=causal_mask)
            input_cast[:, org_steps + step, :] = step_output[:, org_steps + step - 1, :]
            breakpoint()

        op = self.lin_op(input_cast)
        reshaped_op = op.reshape((op.shape[0], op.shape[1], 50, 9)).transpose(1, 2)

        return reshaped_op

    def generate(self, inp, steps=60):
        reshaped_inp = inp.transpose(1, 2).flatten(start_dim=2)

        working_inp = reshaped_inp

        for step in range(steps):
            inp1 = self.lin_inp(working_inp)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                inp1.shape[1]
            ).to(self.config["DEVICE"])
            op1 = self.network(inp1, mask=causal_mask)
            op = self.lin_op(op1)
            working_inp = torch.concat((working_inp, op[:, -1:, :]), dim=1)
            # breakpoint()
            pass

        reshaped_op = working_inp.reshape(
            (working_inp.shape[0], working_inp.shape[1], 50, 9)
        ).transpose(1, 2)

        return reshaped_op
