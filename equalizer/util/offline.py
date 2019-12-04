import os
import numpy as np
import torch
from torch.optim import Adam
from ..channel import LinearChannel
from tqdm import tqdm

def apply_list(fun, *x):
    if len(x) == 1:
        return fun(x[0])
    return tuple(map(fun, x))

def to_torch(x):
    return torch.from_numpy(x.astype(np.float32))

def to_numpy(x):
    return x.detach().cpu().numpy().astype(np.float)

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
    label = label.astype(np.int32)
    data = data / np.sqrt(2)
    return data, label

def gen_ktap(batch_size, pream_size, tap_size, snr, payload_size=0, mod=gen_qpsk, channel=LinearChannel):
    """
    generate ktap data
    if pream_size <= 0 and payload_size > 0, return payload_recv, tap, label
    if payload_size <= 0 and pream_size > 0, return pream, tap, pream_recv
    if both > 0, return pream, pream_recv, payload_recv, label
    """
    ch = channel(tap_size, snr)
    tap = ch.generateParameters(batch_size)

    if pream_size > 0:
        pream, _ = mod(batch_size, pream_size)
        pream_recv = ch.process(tap, pream)
    
    if payload_size > 0:
        payload, label = mod(batch_size, payload_size)
        payload_recv = ch.process(tap, payload)
        
    if payload_size <= 0:
        return pream, tap, pream_recv
    elif pream_size <= 0:
        return payload_recv, tap, label
    else:
        return pream, pream_recv, payload_recv, label

def demod_qpsk(x):
    l = 2 * (x[..., 0] > 0) + 1 * (x[..., 1] > 0)
    l = l.astype(np.int32)
    return l

def bit_error_rate(x, y, bits):
    x = x ^ y
    ret = 0
    for i in range(bits):
        ret += np.mean((x & (1 << i)) != 0)
    return ret / bits

def batch_eval(model, inputs, output, loss_func, batch_size, optim=None, silent=True):
    if optim == None:
        model.eval()
    else:
        model.train()
    tot_size = inputs[0].shape[0]
    r = range(0, tot_size, batch_size)
    if not silent:
        r = tqdm(r)
    loss_tot = 0
    iteration = 0
    for i in r:
        batch_inputs = apply_list(lambda x: x[i:i+batch_size], *inputs)
        batch_output = output[i:i+batch_size]
        if optim != None:
            optim.zero_grad()
        est = model.forward(*batch_inputs)
        loss = loss_func(est, batch_output)
        loss_tot += loss.item()
        iteration += 1
        if optim != None:
            loss.backward()
            optim.step()
    return loss_tot / iteration

def train_e2e(model, inputs, output, loss_func, train_size, batch_size, epoch, save_path, optim=Adam, silent=True):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(save_path):
        print("load existing model")
        model.load_state_dict(torch.load(save_path))

    opt = Adam(model.parameters())
    get_train = lambda x: to_torch(x[:train_size])
    train_inputs = apply_list(get_train, *inputs)
    train_output = apply_list(get_train, output)
    get_test = lambda x: to_torch(x[train_size:])
    test_inputs = apply_list(get_test, *inputs)
    test_output = apply_list(get_test, output)

    train_loss = []
    test_loss = []

    for i in range(epoch):
        train_loss.append(batch_eval(model, train_inputs, train_output, loss_func, batch_size, opt, silent))

        print("test on epoch {}".format(i + 1))
        test_loss.append(batch_eval(model, test_inputs, test_output, loss_func, batch_size))
        
        print("epoch {} train loss: {}, test loss: {}".format(i + 1, train_loss[-1], test_loss[-1]))
        torch.save(model.state_dict(), save_path)
        print('save complete')
    
    return np.array(train_loss), np.array(test_loss)

def eval_e2e(model, pream_size, payload_size, tap_size, snr, eval_size, batch_size=None, silent=True):
    if batch_size == None:
        batch_size = eval_size
    
    batches = eval_size // batch_size
    it = range(batches)
    if not silent:
        it = tqdm(it)
    ber = 0
    for _ in it:
        pream, pream_recv, payload_recv, label = gen_ktap(batch_size, pream_size, tap_size, snr, payload_size)

        model.update_preamble(pream, pream_recv)
        payload_est = model.estimate(payload_recv)

        label_est = demod_qpsk(payload_est)
        ber += bit_error_rate(label_est, label, 2)
    return ber / batches
