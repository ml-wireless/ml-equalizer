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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('algo', nargs='+', help='algorithms')
    args = parser.parse_args()

    bers = [ eval_snr ]
    for model in args.algo:
        print("evaluating", model)
        eval_e2e = lambda snr: offline.eval_e2e(model_bank[model], pream_size, payload_size, model_tap_size, snr, eval_size)
        bers.append(list(np.mean(eval_e2e(snr)) for snr in tqdm(eval_snr)))
    bers = np.array(bers)

    with open(args.output, 'w') as f:
        print('# snr', *args.algo, file=f)
        for i in range(bers.shape[1]):
            print(*bers[:, i], file=f)
