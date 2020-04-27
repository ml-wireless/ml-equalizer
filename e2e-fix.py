from argparse import ArgumentParser
import numpy as np
import equalizer.util.offline as offline
from modelBank import model_bank
from tqdm import tqdm

# common parameters
pream_size = 40
model_tap_size = 2

# eval parameters
eval_size = 10000
payload_size = 200
eval_snr = [-3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
taps = [
    [0.9, 0.1],
    [-0.05, 0.95],
    [0.52, -0.48],
    [-0.6, -0.4],
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('algo', nargs='+', help='algorithms')
    args = parser.parse_args()

    bers = [ eval_snr ]
    labels = []
    for idx, tap in enumerate(taps):
        tap = np.array(tap)
        tap /= np.sqrt(np.sum(tap ** 2))
        print("tap", tap)
        for model in args.algo:
            print("evaluating", model)
            eval_e2e = lambda snr: offline.eval_e2e(model_bank[model], pream_size, payload_size, model_tap_size, snr, eval_size, fixTap=tap)
            bers.append(list(np.mean(eval_e2e(snr)) for snr in tqdm(eval_snr)))
            labels.append(model + '-' + str(idx))
    bers = np.array(bers)

    with open(args.output, 'w') as f:
        print('# snr', *labels, file=f)
        for i in range(bers.shape[1]):
            print(*bers[:, i], file=f)
