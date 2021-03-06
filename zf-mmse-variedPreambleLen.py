import numpy as np
import torch
import equalizer.util.offline as offline
from equalizer.model.classic import ZeroForcing, MMSEInverse, MMSEEstimator, MMSEEqualizer, ClassicTap
from equalizer.model.tap import TapEstimator
from tqdm import tqdm

# common parameters
#pream_size = 40
pream_sizes = [32, 40, 60, 80, 100, 120, 128]
model_tap_size = 2

# model parameters
order = 31
expand = 8192
eps = 0

# eval parameters
eval_size = 10000
payload_size = 200
eval_snr = [1, 10, 50, 100, 500, 1000, 5000]
eval_dat = 'figure/mmse-preLen.dat'

if __name__ == "__main__":
    est = MMSEEstimator(model_tap_size)
    zf = ClassicTap(est, ZeroForcing, expand=expand, trunc=order, eps=eps)
    mmse1 = MMSEInverse(order)

    model = mmse1   

    bers = [ pream_sizes ]
    for snr in eval_snr:
        bers.append(list(offline.eval_e2e(model, pream_size, payload_size, model_tap_size, snr, eval_size) for pream_size in tqdm(pream_sizes)))
    bers = np.array(bers)

    with open(eval_dat, 'w') as f:
        for i in range(bers.shape[1]):
            print(' '.join(map(str, bers[:, i])), file=f)
