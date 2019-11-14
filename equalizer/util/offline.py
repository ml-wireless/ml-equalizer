import numpy as np
import torch
from torch.optim import Adam
from ..channel import LinearChannel
from tqdm import tqdm

def to_torch(x):
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    return x

def gen_qpsk(batch_size, seq_size):
    """
    generate qpsk data
    batch_size: m
    seq_size: n
    returns: data, label
    data: (m, n, 2)
    label: (m, n) in {0, 1, 2, 3}
    """
    data = np.random.choice([-1, 1], size=(batch_size, seq_size, 2))
    label = data[..., 0] + 1 + (data[..., 1] + 1) // 2
    data = data / np.sqrt(2)
    return data, label

def gen_ktap(batch_size, seq_size, tap_size, snr, mod=gen_qpsk, channel=LinearChannel):
    send, label = mod(batch_size, seq_size)
    ch = channel(tap_size, snr)
    param = ch.generateParameters(batch_size)
    recv = ch.process(param, send)
    return send, label, param, recv

def train_model(model, optim, inputs, output, loss_func, batch_size, silent=True):
    model.train()
    loss_tot = 0
    iteration = 0
    train_size = inputs[0].shape[0]
    r = range(0, train_size, batch_size)
    if not silent:
        r = tqdm(r)
    for i in r:
        optim.zero_grad()
        batch = tuple(map(lambda x: x[i:i+batch_size], inputs))
        est = model.forward(*batch)
        loss = loss_func(est, output[i:i+batch_size])
        loss_tot += loss.item()
        loss.backward()
        optim.step()
        iteration += 1
    return loss_tot / iteration

def test_model(model, inputs, output, loss_func):
    model.eval()
    est = model.forward(*inputs)
    loss = loss_func(est, output)
    return loss.item()

def train_e2e(model, inputs, output, loss_func, train_size, batch_size, epoch, save_path, optim=Adam, silent=True):
    opt = Adam(model.parameters())
    get_train = lambda x: to_torch(x[:train_size])
    train_inputs = tuple(map(get_train, inputs))
    train_output = get_train(output)
    get_test = lambda x: to_torch(x[train_size:])
    test_inputs = tuple(map(get_test, inputs))
    test_output = get_test(output)
    train_loss = []
    test_loss = []
    for i in range(epoch):
        train_loss.append(train_model(model, opt, train_inputs, train_output, loss_func, batch_size, silent))
        test_loss.append(test_model(model, test_inputs, test_output, loss_func))
        print("epoch {} train loss: {}, test loss: {}".format(i, train_loss[-1], test_loss[-1]))
    torch.save(model.state_dict(), save_path)
    return np.array(train_loss), np.array(test_loss)
