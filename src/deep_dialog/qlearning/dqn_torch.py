import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_i2h = nn.Linear(self.input_size, self.hidden_size)
        self.linear_h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = F.tanh(self.linear_i2h(x))
        x = self.linear_h2o(x)
        return x

    def predict(self, x):
        y = self.forward(x)
        return torch.argmax(y, 1)

