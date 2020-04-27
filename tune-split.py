from argparse import ArgumentParser
import numpy as np
import equalizer.util.offline as offline
from equalizer.model.classic import FilterEqualizer, ClassicTap
from equalizer.model.tap import HybridLmsEstimator
from modelBank import cnn_est, cnn_est_31
from tqdm import tqdm

pream_size = 40
model_tap_size = 2

models = {
    'hybrid-5': lambda s: ClassicTap(HybridLmsEstimator(cnn_est, s, mu=0.03), FilterEqualizer),
    'hybrid-31': lambda s: ClassicTap(HybridLmsEstimator(cnn_est_31, s, mu=0.01), FilterEqualizer),
}

eval_size = 10000
payload_size = 200
snrs = [3, 6, 9, 12, 15, 18]
splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output file')
    args = parser.parse_args()

    bers = [ splits ]
    keys = []
    for k, v in models.items():
        for snr in snrs:
            key = k + '-snr-' + str(snr)
            print('evaluating', key)
            keys.append(key)
            eval_e2e = lambda s: offline.eval_e2e(v(s), pream_size, payload_size, model_tap_size, snr, eval_size)
            bers.append(list(np.mean(eval_e2e(split)) for split in tqdm(splits)))
    bers = np.array(bers)
    
    with open(args.output, 'w') as f:
        print('# split', *keys, file=f)
        for i in range(bers.shape[1]):
            print(*bers[:, i], file=f)

