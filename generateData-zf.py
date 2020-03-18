import numpy as np
import torch
import pickle
import equalizer.util.offline as offline
from equalizer.channel.linear import inverse_tap_fft
from equalizer.model.classic import ZeroForcing,MMSEEstimator, ClassicTap
from tqdm import tqdm
#from offline import gen_qpsk, gen_ktap

# common parameters
pream_size = 40
tap_size = 5
data_size = 10000
zf_data = 'data/zf_data_snr10_pream40'

# model parameters
order = 5
expand = 8192
eps = 0
snr = 10

if __name__ == "__main__":

    pream, tap, pream_recv = offline.gen_ktap(data_size, pream_size, tap_size, snr)
    inverse = inverse_tap_fft(tap, expand=expand, trunc=order, eps=eps)
    data2save = [pream, pream_recv, inverse]
    # print(inverse.shape)

    with open(zf_data + '.pkl', 'wb') as f:
        pickle.dump(data2save, f)
