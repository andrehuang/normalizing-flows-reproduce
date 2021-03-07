import torch
import torch.nn as nn
import torch.nn.functional as F

# Model function for decoder and encoder

# A simple MLP architecture
class MLP(nn.Module):
    def __init__(
          self,
          input_dim,
          features,
          depth,
          num_outputs=None
    ):
        super().__init__()

        self.num_outputs = num_outputs
        self.input_dim = input_dim
        self.first = nn.Linear(input_dim, features)
        self.layers = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.first(x)

        for layer in self.layers:
            x = F.relu(x)
            x = layer(x)

        if self.num_outputs is not None:
            x = self.last(x)

        return x

def MLP_encoder(args):
    model = MLP(input_dim=784, features=256, depth=3, num_outputs=args.encoder_dim)
    return model

def MLP_decoder(args):
    model = MLP(input_dim=args.z_size, features=256, depth=3, num_outputs=args.decoder_dim)
    return model