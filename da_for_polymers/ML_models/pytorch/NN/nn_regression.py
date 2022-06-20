
from torch import nn
import tensorboard
import torch.nn.functional as F

class NNModel(nn.Module):
    def __init__(self, config):
        """Instantiates NN linear model with arguments from 

        Args:
            config (args): Model Configuration parameters.
        """
        super().__init__()
        self.nonlinear_layer = nn.ReLU
        self.linearlayers = nn.Sequential(nn.Embedding(config["vocab_length"], config.embedding), nn.Linear(config.embedding_size, config.hidden_size), nn.Linear(config.hidden_size, config.output_size))
        self.loss = nn.MSELoss()
    
    def forward(self, x):
        output = F.relu(self.linearlayers(x))
        return output

