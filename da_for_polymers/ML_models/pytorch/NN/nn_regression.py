from typing import OrderedDict
import torch
from torch import nn
import numpy as np
import tensorboard
import torch.nn.functional as F

class OrthoLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

class XavierLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

class NNModel(nn.Module):
    def __init__(self, config):
        """Instantiates NN linear model with arguments from

        Args:
            config (args): Model Configuration parameters.
        """
        super().__init__()
        self.embeds: nn.Sequential = nn.Sequential(
            nn.Embedding(config["vocab_length"], config.embedding_size),
            nn.ReLU(),
            OrthoLinear(config.embedding_size, config.hidden_size),
            nn.ReLU(),
        )
        self.linearlayers: nn.Sequential = nn.Sequential(
            OrderedDict([OrthoLinear(config.hidden_size, config.hidden_size), nn.ReLU() for _ in range(config["n_layers"])])
        )
        self.output: OrthoLinear = OrthoLinear(config.hidden_size, config.output_size)

    def forward(self, x: torch.tensor):
        """
        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        embeds: torch.tensor = self.embeds(x)
        linear: torch.tensor = self.linearlayers(embeds)
        output: torch.tensor = self.output(linear)
        return output
