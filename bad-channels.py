from argparse import ArgumentParser
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import equalizer.util.offline as offline
from modelBank import model_bank

pream_size = 40
model_tap_size = 2
snr = 9

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
    avg = np.mean(bers)

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.title('BER heatmap, SNR={}, avg={:.2%}'.format(snr, avg))
    plt.xlabel('a0')
    plt.ylabel('a1')
    H1, xe, ye = np.histogram2d(taps[..., 0], taps[..., 1], 32, weights=bers)
    H2, _, _ = np.histogram2d(taps[..., 0], taps[..., 1], 32)
    # X, Y = np.meshgrid(xe, ye)
    # plt.pcolormesh(X, Y, H)
    plt.imshow(H1 / H2, extent=[xe[0], xe[-1], ye[0], ye[-1]])
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.colorbar()

    plt.subplot(122)
    ang = np.arctan2(taps[..., 1], taps[..., 0])
    plt.title('BER vs argument, SNR={}, avg={:.2%}'.format(snr, avg))
    plt.xlabel('arg(a0, a1)')
    plt.xticks((-np.pi, -np.pi/2, 0, np.pi/2, np.pi), ('$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'))
    plt.ylabel('BER')
    plt.scatter(ang, bers, s=2)

    plt.savefig(args.output)
