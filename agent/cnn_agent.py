import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.operators = nn.ModuleList([
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
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Dropout(.1),
            nn.Linear(50, 5)
        ])

    def forward(self, x):
        for operator in self.operators:
            x = operator(x)

        return x

    def load(self, path="cnn.pt"):
        pass


class CnnAgent(object):

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.action_encoder = OneHotEncoder(range(5))

        self.net = Net()
        self.target_net = Net()
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

    def add_experience(self, experience):

        self.experience.append(experience)

    def train(self, batch_size=32):

        s, a, r, s_, t = [], [], [], [], []

        indices = np.random.choice(list(set(range(len(self.experience)))), batch_size)

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

        Q_ = self.net(s_)
        Q = r + 0.9 * Q_ * (1 - t)

        self.optimizer.zero_grad()

        Qpred = self.net(s)

        loss = self.loss(Qpred * a, Q * a)
        loss.backward()
        self.optimizer.step()


    def load(self, path):
        if os.path.isfile(path):
            self.net.load_state_dict(torch.load(path))
        else:
            raise ValueError(f"No model found at {path}")

    def save(self, path):
        torch.save(self.net.state_dict(), path)
