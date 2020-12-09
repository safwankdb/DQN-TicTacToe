REPLAY_SIZE = 50_000
WARMUP_SIZE = 5_000
GAMMA = 0.9
TARGET_UPDATE = 10
BATCH_SIZE = 32

EPS_START = 1
DECAY_LEN = 5_000
EPS_END = 0.1
SAVE_EVERY = 4_000



from torch import nn

class ScaleLayer(nn.Module):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        self.alpha = 1/(1-GAMMA)

    def forward(self, x):
        return self.alpha * x


def create_model(ins, outs):
    model = nn.Sequential(
        nn.Linear(ins, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 9),
    )
    return model
