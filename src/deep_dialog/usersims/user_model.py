import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class SimulatorModel(nn.Module):
    def __init__(self,
                 agent_action_size,
                 hidden_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1):
        super(SimulatorModel, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)
        self.linear_h2a = nn.Linear(hidden_size, user_action_size)

    def forward(self, s, a):
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze(1)
        h = F.tanh(h_s + h_a)

        reward = self.linear_h2r(h)
        term = self.linear_h2t(h)
        action = F.log_softmax(self.linear_h2a(h), 1)

        return reward, term, action

    def predict(self, s, a):
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze(1)
        h = F.tanh(h_s + h_a)

        reward = self.linear_h2r(h)
        term = F.sigmoid(self.linear_h2t(h))
        action = F.log_softmax(self.linear_h2a(h), 1)

        return reward, term, action.argmax(1)
