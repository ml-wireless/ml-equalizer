import numpy as np
import torch
import equalizer.util.offline as offline
from equalizer.model.classic import ZeroForcing, MMSE1
from equalizer.model.tap import TapEstimator
from tqdm import tqdm

# common parameters
pream_size = 40
model_tap_size = 2
est_path = 'model/tap2-est.bin'

# model parameters
order = 5
expand = 32
eps = 0.01

# eval parameters
eval_size = 10000
payload_size = 200
eval_snr = [1, 10, 50, 100, 500, 1000, 5000]
eval_dat = 'figure/zf-mmse-ber.dat'

if __name__ == "__main__":
    est = TapEstimator(pream_size, model_tap_size)
    est.load_state_dict(torch.load(est_path))
    zf = ZeroForcing(est, expand=expand, trunc=order, eps=eps)
    mmse = MMSE1(order)

    models = [zf, mmse]
    bers = [ eval_snr ]
    for model in models:
        bers.append(list(offline.eval_e2e(zf, pream_size, payload_size, model_tap_size, snr, eval_size) for snr in tqdm(eval_snr)))
    bers = np.array(bers)

    with open(eval_dat, 'w') as f:
        for i in range(bers.shape[1]):
            print(' '.join(map(str, bers[:, i])), file=f)
