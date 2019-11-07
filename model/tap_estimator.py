import numpy as np
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

if __name__ == "__main__":
    import os.path as path
    import sys
    sys.path.append('.')
    from channel import LinearChannel

    m = 50000
    n = 50

    # quick QPSK data
    data = np.random.choice([-1, 1], size=(m, n, 2))
    label = data[..., 0] + 1 + (data[..., 1] + 1) // 2
    data = data / np.sqrt(2)

    tap_size = 2
    snr = 20
    channel = LinearChannel(tap_size, snr)
    param = channel.generateParameters(m)
    recv_data = channel.process(param, data)

    def test(model, data, param, recv_data):
        model.eval()
        est = model.forward(data, recv_data)
        loss = F.mse_loss(est, param)
        return loss.detach()

    def train(model, optim, data, param, recv_data, batch=200):
        model.train()
        loss_tot = 0
        for i in range(n, data.shape[0], batch):
            optim.zero_grad()
            est = model.forward(data[i:i+batch], recv_data[i:i+batch])
            loss = F.mse_loss(est, param[i:i+batch])
            loss_tot += loss.detach()
            loss.backward()
            optim.step()
        return loss_tot / (data.shape[0] // batch)
    
    train_size = 40000
    model = TapEstimator(n, tap_size)
    from torch.optim import Adam
    optim = Adam(model.parameters())
    train_data = tuple(torch.from_numpy(x[:train_size]).float() for x in (data, param, recv_data))
    test_data = tuple(torch.from_numpy(x[train_size:]).float() for x in (data, param, recv_data))
    for i in range(50):
        train_loss = train(model, optim, *train_data)
        test_loss = test(model, *test_data)
        print("epoch {} train loss: {}, test loss: {}".format(i, train_loss, test_loss))
