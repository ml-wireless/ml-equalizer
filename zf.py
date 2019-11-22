import numpy as np
import torch
import equalizer.util.offline as offline
from equalizer.model.classic import ZeroForcing
from equalizer.model.tap import TapEstimator
from tqdm import tqdm

# common parameters
pream_size = 40
model_tap_size = 2
est_path = 'model/tap2-est.bin'

# model parameters
inverse_tap_size = 5
expand = 32
eps = 0.01

# eval parameters
eval_size = 10000
payload_size = 200
eval_snr = [1, 10, 50, 100, 500, 1000, 5000]
eval_dat = 'figure/zf-est-ber.dat'

def eval_e2e(model, snr):
    pream, pream_recv, payload_recv, label = offline.gen_ktap(eval_size, pream_size, model_tap_size, snr, payload_size)
    model.update_preamble(pream, pream_recv)
    payload_est = model.estimate(payload_recv)
    label_est = offline.demod_qpsk(payload_est)
    return offline.bit_error_rate(label_est, label, 2)

if __name__ == "__main__":
    est = TapEstimator(pream_size, model_tap_size)
    est.load_state_dict(torch.load(est_path))
    model = ZeroForcing(est, expand=expand, trunc=inverse_tap_size, eps=eps)
    
    bers = [ eval_e2e(model, snr) for snr in tqdm(eval_snr) ]
    with open(eval_dat, 'w') as f:
            for snr, ber in zip(eval_snr, bers):
                print('{} {}'.format(snr, ber), file=f)
