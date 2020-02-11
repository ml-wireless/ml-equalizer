import sys
sys.path.append('../')
import equalizer.util.offline as offline
from equalizer.channel.linear import im_tap, tap_proc
import numpy as np

#use lms to calculate inverse channel weights
def lms(pream_recv, pream, order, mu=0.1):
    # init weights to random complex values
    # (..., order)
    w_size = pream.shape[:-1] + (order,)
    w = np.random.normal(size=w_size) + 1j * np.random.normal(size=w_size)

    # (..., length)
    e = np.zeros(pream_recv.shape, dtype=np.complex_)
    left = (order - 1) // 2
    pad_size = ((0, 0),) * (pream.ndim - 1) + ((left, order - 1 - left),)
    pream_recv = np.pad(pream_recv, pad_size, 'constant')

    for i in range(pream.shape[-1]):
        x = pream_recv[..., i:i+order]
        # apply the FIR to get current output
        y = np.einsum('...i,...i->...', x, w)
        # compute latest error
        e[..., i] = pream[..., i] - y # cost / error
        # update weights
        w += mu * e[..., i:i+1] * x.conj()

    return w, e

#estimate original signal using recieved signal
def predict(signal, w, order):
    # init empty array for equalized symbols
    # y = 0j*np.zeros(signal.shape[0])
    # for i in range(order-1,signal.shape[0]):
    #     y[i] = signal[(i - (order - 1)):(i + 1)].T @ w

    #np.convolve flips slider
    return tap_proc(im_tap(np.flip(w, axis=-1)), signal)

class lms_model(object):
    def __init__(self, order):
        self.order = order

    def inverse_channel(self, pream, pream_recv, mu=0.1):
        self.w, e = lms(pream_recv, pream, self.order, mu)
        # returning error so consuming module can analyze LMS
        # performance
        return e

    def get_inverse_channel(self):
        return self.w

    def estimate(self, signal):
        return predict(signal, self.w, self.order)
