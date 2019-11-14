import torch
from torch import nn
import torch.nn.functional as F

class TapEstimator(nn.Module):
    def __init__(self, seq_size, tap_size, middles=(300, 300)):
        super(TapEstimator, self).__init__()
        # linear layers
        self.fcs = []
        middles = (seq_size * 4,) + middles + (tap_size,)
        for i in range(len(middles) - 1):
            fc = nn.Linear(middles[i], middles[i + 1])
            self.add_module('fc' + str(i), fc)
            self.fcs.append(fc)
    
    def forward(self, send, recv):
        """
        send: (m, n, 2), m is batch size
        recv: (m, n, 2)
        returns: (m, k), k is tap number
        """
        x = torch.cat((send, recv), dim=-1)
        x = x.view(-1, x.shape[-1] * x.shape[-2])
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i == len(self.fcs) - 1:
                x = torch.tanh(x)
            else:
                x = torch.sigmoid(x)
        return x

class TapEqualizer(nn.Module):
    def __init__(self, tap_size, hidden_size=45, dense_size=100, layers=2):
        super(TapEqualizer, self).__init__()
        self.rnn = nn.LSTM(2 + tap_size, hidden_size, layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size, 2)
    
    def forward(self, tap, recv):
        """
        tap: (m, k), k is tap_size, m is batch size
        recv: (m, n, 2)
        returns: (m, n, 2)
        """
        tap = torch.unsqueeze(tap, dim=-2)
        repeats = (-1,) * (tap.ndim - 2) + (recv.shape[-2], -1)
        tap = tap.expand(repeats)
        x = torch.cat((recv, tap), dim=-1)
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
