import numpy as np
from ..util import offline
from ..channel.linear import inverse_tap_fft, tap_proc, real_tap, im_tap

class ZeroForcing(object):
    def __init__(self, algo=inverse_tap_fft, **params):
        self.algo = lambda a: algo(a, **params)
    
    def update_tap(self, tap):
        self.inv = real_tap(self.algo(tap))
    
    def estimate(self, recv):
        return tap_proc(self.inv, recv)

def mmse1(recv, pream, order, eps=0):
    """
    mmse with unknown channel
    recv: received signal, (*, k, 2)
    pream: sent signal, (*, k, 2)
    order: order of the filter
    returns: (*, order), the inverse filter

    NOTE: assume real tap
    """
    temp_pream = pream[..., 0] + pream[..., 1] * 1j
    temp_recv = recv[..., 0] + recv[..., 1] * 1j
    rpad = (order - 1) // 2
    lpad = order - 1 - rpad
    temp_recv = np.pad(temp_recv, ((0, 0),) * (temp_recv.ndim - 1) + ((lpad, rpad),), 'constant')

    R = np.zeros(temp_recv.shape[:-1] + (order, order), dtype=np.complex_)
    d = np.zeros(temp_recv.shape[:-1] + (order,), dtype=np.complex_)
    for i in range(recv.shape[-2]):
        Ri = temp_recv[..., i:i+order]
        R += np.real(np.expand_dims(Ri, -2).conj() * np.expand_dims(Ri, -1))
        d += np.real(temp_pream[..., i:i+1] * Ri.conj())
    return np.flip(np.linalg.solve(R + np.eye(order) * eps, d), -1)

class MMSEInverse(object):
    def __init__(self, order, algo=mmse1):
        self.algo = lambda s, r: real_tap(algo(r, s, order))
    
    def update_preamble(self, pream, pream_recv):
        self.inv = self.algo(pream, pream_recv)

    def estimate(self, recv):
        return tap_proc(self.inv, recv)

class MMSEEstimator(object):
    def __init__(self, order, algo=mmse1):
        self.algo = lambda s, r: algo(r, s, order)
    
    def estimate_tap(self, pream, pream_recv):
        return self.algo(pream_recv, pream)

# mmse with known channel
def mmse2(recv, tap, eps=0.01):
    temp_recv = (recv[..., 0] + recv[..., 1]*1j)
    recv_len = temp_recv.shape[-1]
    H = np.zeros(temp_recv.shape[:-1] + (recv_len, recv_len), dtype=tap.dtype)
    tap = np.flip(tap, axis=-1)
    tap_len = tap.shape[-1]
    for i in range(recv_len):
        st = i - tap_len + tap_len // 2
        ed = st + tap_len
        st1 = max(0, st)
        ed1 = min(recv_len, ed)
        H[..., i, st1:ed1] = tap[..., st1-st:tap_len+ed1-ed]
    A = np.einsum('...ki,...kj->...ij', H, H.conj()) + eps * np.eye(recv_len)
    b = np.einsum('...ji,...j->...i', H.conj(), temp_recv)
    ret = np.linalg.solve(A, b)
    return ret

class MMSEEqualizer(object):
    def __init__(self, algo=mmse2):
        self.algo = algo

    def update_tap(self, tap):
        self.tap = tap
    
    def estimate(self, recv):
        return self.algo(recv, self.tap)

class ClassicTap(object):
    def __init__(self, est, eq, **params):
        self.est = est
        self.eq = eq(**params)
    
    def update_preamble(self, pream, pream_recv):
        tap = self.est.estimate_tap(pream, pream_recv)
        self.eq.update_tap(tap)
    
    def estimate(self, recv):
        return self.eq.estimate(recv)

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

class LMS(object):
    def __init__(self, order, algo=lms):
        self.algo = lambda s, r: algo(offline.to_complex(r), offline.to_complex(s), order)
    
    def update_preamble(self, pream, pream_recv):
        w, e = self.algo(pream, pream_recv)
        self.errors = e
        self.inv = im_tap(np.flip(w, axis=-1))
    
    def estimate(self, recv):
        return tap_proc(self.inv, recv)
