import torch
from torch import nn
import torch.nn.functional as F
from ..util import offline

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
                x = torch.relu(x)
        return x
    
    def estimate_tap(self, pream, pream_recv):
        self.eval()
        pream, pream_recv = offline.apply_list(offline.to_torch, pream, pream_recv)
        return offline.to_numpy(self.forward(pream, pream_recv))

class TapEqualizer(nn.Module):
    def __init__(self, tap_size, hidden_size=45, dense_size=200, layers=2):
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

class CNNEstimator(nn.Module):
    def __init__(self, tap_size):
        super(CNNEstimator, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, tap_size)
    
    def forward(self, send, recv):
        x = torch.cat((send, recv), dim=-1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool1d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.mean(x, -1)
        x = self.fc(x)
        x = F.tanh(x)
        return x

class NeuralTap(object):
    def __init__(self, estimator, equalizer):
        self.estimator = estimator
        self.equalizer = equalizer
        
    def update_preamble(self, pream, pream_recv):
        self.estimator.eval()
        pream, pream_recv = offline.apply_list(offline.to_torch, pream, pream_recv)
        self.param_est = self.estimator.forward(pream, pream_recv)
    
    def estimate(self, recv):
        self.equalizer.eval()
        return offline.to_numpy(self.equalizer.forward(self.param_est, offline.to_torch(recv)))
