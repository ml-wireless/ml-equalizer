from argparse import ArgumentParser
import numpy as np
import torch
import equalizer.util.offline as offline
from equalizer.model.classic import ZeroForcingEstimator, MMSEEstimator, LMSEstimator, FilterEqualizer, ClassicTap
from equalizer.model.tap import TapEstimator, CNNEstimator, HybridLmsEstimator
from tqdm import tqdm

# common parameters
pream_size = 40
model_tap_size = 2

# model parameters
order = 5
order_lms = 5
expand = 8192
eps = 0

# cnn parameters
order_cnn = 5
path_cnn = 'model/cnn-est-zf.bin'

# eval parameters
eval_size = 10000
payload_size = 200
eval_snr = [-3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

if __name__ == "__main__":
    zf_est = ZeroForcingEstimator(expand=expand, trunc=order, eps=eps)
    mmse_est = MMSEEstimator(order, True)
    cnn_est = CNNEstimator(order_cnn)
    cnn_est.load_state_dict(torch.load(path_cnn))
    hybrid_est = HybridLmsEstimator(cnn_est, order_cnn, 0.5)
    lms_est = LMSEstimator(order_lms)

    model_bank = {
        "zf": ClassicTap(zf_est, FilterEqualizer),
        "mmse": ClassicTap(mmse_est, FilterEqualizer),
        "lms": ClassicTap(lms_est, FilterEqualizer),
        "cnn": ClassicTap(cnn_est, FilterEqualizer),
        "hybrid": ClassicTap(hybrid_est, FilterEqualizer),
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
