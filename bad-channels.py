from argparse import ArgumentParser
import numpy as np
import equalizer.util.offline as offline
from modelBank import model_bank

pream_size = 40
model_tap_size = 2
snr = 20

eval_size = 10000
payload_size = 200

badThres = 0.05

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('algo', help='algorithms')
    args = parser.parse_args()

    model = model_bank[args.algo]
    bers, taps = offline.eval_e2e(model, pream_size, payload_size, model_tap_size, snr, eval_size, retTap=True)
    bads = taps[bers > badThres]

    with open(args.output, 'w') as f:
        for i in range(bads.shape[0]):
            print(' '.join(map(str, bads[i])), file=f)
