import numpy as np
from ..util import offline
from ..channel.linear import inverse_tap_fft, tap_proc, real_tap, im_tap

class ZeroForcingEqualizer(object):
    def __init__(self, algo=inverse_tap_fft, **params):
        self.algo = lambda a: algo(a, **params)
    
    def update_tap(self, tap):
        self.inv = real_tap(self.algo(tap))
    
    def estimate(self, recv):
        return tap_proc(self.inv, recv)

class ZeroForcing(object):
    def __init__(self, est, eq=ZeroForcingEqualizer, **params):
        self.est = est
        self.eq = eq(**params)
    
    def update_preamble(self, pream, pream_recv):
        self.est.eval()
        pream, pream_recv = offline.apply_list(offline.to_torch, pream, pream_recv)
        self.eq.update_tap(offline.to_numpy(self.est.forward(pream, pream_recv)))
    
    def estimate(self, recv):
        return self.eq.estimate(recv)

def mmse1(recv, pream, order):
    """
    mmse with unknown channel
    recv: received signal, (*, k, 2)
    pream: sent signal, (*, k, 2)
    order: order of the filter
    returns: (*, 2 * order + 1) dtype=np.complex_, the inverse filter
    """
    temp_pream = pream[..., 0] + pream[..., 1] * 1j
    temp_recv = recv[..., 0] + recv[..., 1] * 1j
    temp_recv = np.pad(temp_recv, ((0, 0),) * (temp_recv.ndim - 1) + ((order, order),), 'constant')

    R = np.zeros(temp_recv.shape[:-1] + (2 * order + 1, 2 * order + 1), dtype=np.complex_)
    d = np.zeros(temp_recv.shape[:-1] + (2 * order + 1,), dtype=np.complex_)
    for i in range(order, recv.shape[1]+order):
        Ri = temp_recv[..., i-order:i+order+1]
        R += np.expand_dims(Ri, -2) * np.expand_dims(Ri, -1)
        d += temp_pream[..., i-order:i-order+1] * Ri
    return np.flip(np.linalg.solve(R, d), -1)

class MMSE1(object):
    def __init__(self, order, algo=mmse1):
        self.algo = lambda s, r: im_tap(algo(r, s, order))
    
    def update_preamble(self, pream, pream_recv):
        self.inv = self.algo(pream, pream_recv)

    def estimate(self, recv):
        return tap_proc(self.inv, recv)
