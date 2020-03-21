import numpy as np
import torch
import gzip
import pickle
import equalizer.util.offline as offline
from equalizer.model.classic import inverse_tap_fft
from tqdm import tqdm
#from offline import gen_qpsk, gen_ktap

# common parameters
pream_size = 40
tap_size = 2
data_size = 20000
zf_data = 'data/zf_data_snr10_pream40'

# model parameters
order = 5
expand = 8192
eps = 0
snr = 10

if __name__ == "__main__":
    pream, tap, pream_recv = offline.gen_ktap(data_size, pream_size, tap_size, snr)
    inverse = inverse_tap_fft(tap, expand=expand, trunc=order, eps=eps)
    data2save = {
        'pream': pream,
        'pream_recv': pream_recv,
        'inverse_weights': inverse,
        'gen_taps': tap,
    }

    with gzip.open(zf_data + '.gz', 'wb') as f:
        pickle.dump(data2save, f)
