import torch 
import math
import torch.nn.functional as F
import numpy as np
from torch import nn
class RNN(torch.nn.Module):
    """
    Basic RNN block. This represents a single layer of RNN
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden_state):
        hidden_state = torch.tanh(self.h2h(hidden_state) + self.i2h(input))
        out = self.h2o(hidden_state)
        return out, hidden_state
    
    def init_zero_hidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size)
    
