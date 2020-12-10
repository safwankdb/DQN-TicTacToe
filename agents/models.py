import torch
import torch.nn as nn
import random
from tqdm import tqdm
import numpy as np

from config import create_model
from config import REPLAY_SIZE, WARMUP_SIZE, GAMMA, TARGET_UPDATE, BATCH_SIZE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nRunning on {device.upper()}\n")


class ExperienceReplay:

    def __init__(self, size=REPLAY_SIZE):
        self.size = size
        self.buffer = []
        self.pointer = 0

    def push(self, s, a, r, s_, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pointer] = (s, a, r, s_, done)
        self.pointer = (self.pointer + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_memory = ExperienceReplay()
        self.model = create_model(self.n_states, n_actions).to(device)
        print(self.model)
        self.target_model = create_model(self.n_states, n_actions).to(device)
        self.target_model.eval()
        self.opt = torch.optim.RMSprop(self.model.parameters(), 2e-4)
        # self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
        self.target_counter = 0

    def memorize(self, s, a, r, s_, done=False):
        self.replay_memory.push(s, a, r, s_, done)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.Tensor(x).unsqueeze(0).to(device)
            out = self.model(x)[0].cpu()
        return out

    def forward(self, x):
        out = self.model(x)
        return out

    def train(self, terminal=False):
        if len(self.replay_memory.buffer) < WARMUP_SIZE:
            return
        minibatch = self.replay_memory.sample(BATCH_SIZE)
        S = torch.Tensor([i[0] for i in minibatch]).to(device)
        S_ = torch.Tensor([i[3] for i in minibatch]).to(device)

        self.model.eval()
        with torch.no_grad():
            Q = self.model(S)
            Q_ = self.target_model(S_)

        X = []
        y = []
        A = []
        for i, (s, a, r, _, done) in enumerate(minibatch):
            if done:
                q_ = r
            else:
                q_ = r + GAMMA * Q_[i].max()
            q = Q[i]
            q[a] = q_
            X.append(s)
            y.append(q[a])
            A.append(a)

        X = torch.Tensor(X).to(device)
        y = torch.stack(y).view(-1, 1).to(device)
        A = torch.Tensor(A).long().to(device)

        self.model.train()
        out = self.model(X)
        out = torch.gather(out, 1, A.unsqueeze(-1))
        loss = self.loss(out, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if terminal:
            self.target_counter += 1
            if self.target_counter >= TARGET_UPDATE:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_counter = 0
