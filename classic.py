from argparse import ArgumentParser
import numpy as np
import torch
import equalizer.util.offline as offline
from equalizer.model.classic import ZeroForcing, MMSEInverse, MMSEEstimator, MMSEEqualizer, ClassicTap, LMS, FilterEqualizer
from equalizer.model.tap import TapEstimator, CNNEstimator, HybridLmsEstimator
from tqdm import tqdm

# common parameters
pream_size = 40
model_tap_size = 2

order_cnn = 5
path_cnn = 'model/cnn-est.bin'

# model parameters
order = 5
expand = 8192
eps = 0
lms_order = 3

# eval parameters
eval_size = 10000
payload_size = 200
eval_snr = [1, 10, 50, 100, 500, 1000, 5000]

if __name__ == "__main__":
    est = MMSEEstimator(model_tap_size)
    cnn = CNNEstimator(order_cnn)
    cnn.load_state_dict(torch.load(path_cnn))
    hybrid = HybridLmsEstimator(cnn, order_cnn, 0.5)
    model_bank = {
        "zf": ClassicTap(est, ZeroForcing, expand=expand, trunc=order, eps=eps),
        "mmse": MMSEInverse(order),
        "mmse2": ClassicTap(est, MMSEEqualizer),
        "lms": LMS(lms_order),
        "cnn": ClassicTap(cnn, FilterEqualizer),
        "hybrid": ClassicTap(hybrid, FilterEqualizer),
    }
    
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('algo', nargs='+', help='algorithms')
    args = parser.parse_args()

    bers = [ eval_snr ]
    for model in args.algo:
        print("evaluating", model)
        bers.append(list(offline.eval_e2e(model_bank[model], pream_size, payload_size, model_tap_size, snr, eval_size) for snr in tqdm(eval_snr)))
    bers = np.array(bers)

    with open(args.output, 'w') as f:
        for i in range(bers.shape[1]):
            print(' '.join(map(str, bers[:, i])), file=f)
