import sys
sys.path.append('../')
import equalizer.util.offline as offline
import numpy as np

#use lms to calculate inverse channel weights
def lms(pream_recv, pream, order, mu=0.1):

    # init weights to random complex values
    w = np.array(np.random.normal(0,1,order)) + 1j*np.array(np.random.normal(0,1,order))

    left = (order - 1) // 2
    pream_recv = np.pad(pream_recv, (left, order - 1 - left), 'constant')

    e = np.zeros(pream.shape[0], dtype=np.complex_)

    for i in range(pream.shape[0]):
        x = pream_recv[i:i+order]
        # apply the FIR to get current output
        y = x.T @ w
        # compute latest error
        e[i] = pream[i] - y # cost / error
        # update weights
        w = w + mu*e[i]*x.conj()

    return w, e

#estimate original signal using recieved signal
def predict(signal, w, order):
    # init empty array for equalized symbols
    # y = 0j*np.zeros(signal.shape[0])
    # for i in range(order-1,signal.shape[0]):
    #     y[i] = signal[(i - (order - 1)):(i + 1)].T @ w

    #np.convolve flips slider
    return np.convolve(signal, np.flip(w), mode='same')


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
