import numpy as np
import os
import logging
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class DuelingDQN(nn.Module):

    def __init__(self):
        super(DuelingDQN, self).__init__()

        self.logger = logging.getLogger("dueling_dqn")

        self.logger.setLevel(logging.INFO)

        self.head = nn.ModuleList([
            nn.Conv2d(3, 30, (3, 3)),
            nn.Tanh(),
            nn.Dropout(.1),
            nn.Conv2d(30, 20, (3, 3)),
            nn.Tanh(),
            nn.Dropout(.1),
            nn.Conv2d(20, 10, (3, 3)),
            nn.Tanh(),
            nn.Dropout(.1),
            Flatten(),
            nn.Linear(160, 100),
            nn.Tanh(),
            nn.Dropout(.1),
        ])

        self.state_value_tail = nn.ModuleList([
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Dropout(.1),
            nn.Linear(50, 1)
        ])

        self.action_value_tail = nn.ModuleList([
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Dropout(.1),
            nn.Linear(50, 5)
        ])

    def forward(self, x):

        self.logger.debug("\nHead")
        self.logger.debug(x.shape)

        for operator in self.head:
            x = operator(x)
            self.logger.debug(x.shape)

        state_value = x

        self.logger.debug("\nState Value Tail")
        for operator in self.state_value_tail:
            state_value = operator(state_value)
            self.logger.debug(state_value.shape)

        action_value = x

        self.logger.debug("\nAction Value Tail")
        for operator in self.action_value_tail:
            action_value = operator(action_value)
            self.logger.debug(action_value.shape)

        x = state_value + action_value

        self.logger.debug("\nOutput")
        self.logger.debug(x.shape)

        return x


class CnnAgent(object):

    def __init__(self, input_dim, output_dim, memory_size):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_size = memory_size

        self.action_encoder = OneHotEncoder(range(5))

        self.net = DuelingDQN()
        self.target_net = DuelingDQN()
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.net.to(device)
        self.target_net.to(device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), 0.0001,
                                          amsgrad=True)
        self.loss = nn.MSELoss()

        self.experience = []

    def predict(self, state):

        state = torch.Tensor(state).unsqueeze(0).to(device)

        prediction = self.net(state).cpu().detach()

        action = np.argmax(prediction)

        return action

    def add_experiences(self, experiences):

        assert type(experiences) is list

        for experience in experiences:
            self.experience.append(experience)

        self.experience = self.experience[:self.memory_size]

    def train(self, batch_size=32):

        s, a, r, s_, t = [], [], [], [], []

        indices = np.random.choice(list(set(range(len(self.experience)))),
                                   batch_size)

        for index in indices:

            sample = self.experience[index]


            s.append(sample["s"])
            a.append(sample["a"])
            r.append(sample["r"])
            s_.append(sample["s_"])
            t.append(sample["t"])

        s = np.array(s)
        a = np.array(a)
        r = np.array(r)
        s_ = np.array(s_)
        t = np.array(t, dtype=int)

        a_oh = np.zeros((len(a), 5))

        a_oh[np.arange(len(a)), a] = 1

        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a_oh, dtype=torch.float32).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(-1).to(device)
        s_ = torch.tensor(s_, dtype=torch.float32).to(device)
        t = torch.tensor(t, dtype=torch.float32).unsqueeze(-1).to(device)

        Q_ = self.target_net(s_)
        Q = r + 0.9 * Q_ * (1 - t)

        self.optimizer.zero_grad()

        Qpred = self.net(s)

        loss = self.loss(Qpred * a, Q * a)
        loss.backward()
        self.optimizer.step()

        self.soft_update_target(0.001)

    def load(self, path):
        if os.path.isfile(path):
            self.net.load_state_dict(torch.load(path))
        else:
            raise ValueError(f"No model found at {path}")

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def soft_update_target(self, tau=0.):
        for target_net_param, net_param in zip(self.target_net.parameters(),
                                               self.net.parameters()):
            target_net_param.data.copy_(
                tau * net_param.data + (1.0 - tau) * target_net_param.data)


if __name__ == "__main__":
    net = DuelingDQN()

    s = torch.tensor(np.random.rand(1, 3, 10, 10), dtype=torch.float32).to(
        device)

    net(s)
