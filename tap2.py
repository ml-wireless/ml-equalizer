from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import TapEstimator, TapEqualizer, NeuralTap
from tqdm import tqdm

# common parameters
pream_size = 40
model_tap_size = 2
batch_size = 200
est_path = 'model/tap2-est.bin'
eq_path = 'model/tap2-eq.bin'

# train parameters
data_size = 100000
train_snr = 10
train_size = 80000

# eval parameters
eval_size = 10000
payload_size = 200
eval_tap_size = 2
eval_snr = [1, 10, 50, 100, 500, 1000, 5000]
eval_dat = 'figure/tap2-ber.dat'

estimator = TapEstimator(pream_size, model_tap_size)
equalizer = TapEqualizer(model_tap_size)

def gen_train_data():
    return offline.gen_ktap(data_size, pream_size, model_tap_size, train_snr)

def train_est(epoch):
    pream, tap, recv = gen_train_data()
    return offline.train_e2e(estimator, (pream, recv), tap, F.mse_loss, train_size, batch_size, epoch, est_path)

def train_eq(epoch):
    pream, tap, recv = gen_train_data()
    return offline.train_e2e(equalizer, (tap, recv), pream, F.mse_loss, train_size, batch_size, epoch, eq_path, silent=False)

def eval_e2e():
    estimator.load_state_dict(torch.load(est_path))
    equalizer.load_state_dict(torch.load(eq_path))
    model = NeuralTap(estimator, equalizer)
    bers = []

    for snr in eval_snr:
        batches = eval_size // batch_size
        ber = 0
        for _ in tqdm(range(batches)):
            pream, _, pream_recv, payload_recv, label = offline.gen_ktap(batch_size, pream_size, eval_tap_size, snr, payload_size)

            model.update_preamble(pream, pream_recv)
            payload_est = model.estimate(payload_recv)

            label_est = offline.demod_qpsk(payload_est)
            ber += offline.bit_error_rate(label_est, label, 2)
        bers.append(ber / batches)
    
    return bers

if __name__ == "__main__":
    parser = ArgumentParser('tap2')
    parser.add_argument('action', choices=('eval', 'train_est', 'train_eq'))
    parser.add_argument('epoch', nargs='?', type=int, default=10)
    args = parser.parse_args()
    if args.action == 'eval':
        bers = eval_e2e()
        with open(eval_dat, 'w') as f:
            for snr, ber in zip(eval_snr, bers):
                print('{} {}'.format(snr, ber), file=f)
    elif args.action == 'train_est':
        train_est(args.epoch)
    elif args.action == 'train_eq':
        train_eq(args.epoch)
