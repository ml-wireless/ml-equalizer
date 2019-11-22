import numpy as np
from ..util import offline
from ..channel.linear import inverse_tap_fft, tap_proc, real_tap

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
