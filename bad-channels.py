from argparse import ArgumentParser
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import equalizer.util.offline as offline
from modelBank import model_bank

pream_size = 40
model_tap_size = 2
snrs = [3, 9, 15]

eval_size = 10000
payload_size = 200

badThres = 0.05

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('algo', help='algorithms')
    args = parser.parse_args()

    plt.figure(figsize=(12,5 * len(snrs)))
    for idx, snr in enumerate(snrs):
        model = model_bank[args.algo]
        bers, taps = offline.eval_e2e(model, pream_size, payload_size, model_tap_size, snr, eval_size, retTap=True)
        avg = np.mean(bers)
        
        plt.subplot(len(snrs) * 100 + 21 + idx * 2)
        plt.title('BER heatmap, 2 tap, order=31, SNR={}, avg={:.2%}'.format(snr, avg))
        plt.xlabel('a0')
        plt.ylabel('a1')
        H1, xe, ye = np.histogram2d(taps[..., 0], taps[..., 1], 32, weights=bers)
        H2, _, _ = np.histogram2d(taps[..., 0], taps[..., 1], 32)
        # X, Y = np.meshgrid(xe, ye)
        # plt.pcolormesh(X, Y, H)
        plt.imshow(H1 / H2, extent=[xe[0], xe[-1], ye[0], ye[-1]])
        plt.scatter([0.99, 0.73], [0.11, -0.68], color='red')
        plt.annotate("[0.99, 0.11]", (0.90, 0.11), horizontalalignment='right')
        plt.annotate("[0.73, -0.68]", (0.64, -0.68), horizontalalignment='right')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.colorbar()

        plt.subplot(len(snrs) * 100 + 22 + idx * 2)
        ang = np.arctan2(taps[..., 1], taps[..., 0])
        plt.title('BER vs argument, 2 tap, order=31, SNR={}, avg={:.2%}'.format(snr, avg))
        plt.xlabel('arg(a0, a1)')
        plt.ylim(-0.01, 0.51)
        plt.annotate("[0.99, 0.11]", (np.arctan2(0.11, 0.99) + 0.05, 0.3))
        plt.annotate("[-0.68, 0.73]", (np.arctan2(-0.68, 0.73) - 0.05, 0.3), horizontalalignment='right')
        plt.xticks((-np.pi, -np.pi/2, 0, np.pi/2, np.pi), ('$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'))
        plt.ylabel('BER')
        plt.scatter(ang, bers, s=2)
        plt.axvline(x=np.arctan2(0.11, 0.99), color='red')
        plt.axvline(x=np.arctan2(-0.68, 0.73), color='red')

    plt.savefig(args.output)
