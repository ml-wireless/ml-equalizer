from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import TapEstimator, TapEqualizer, NeuralTap
from tqdm import tqdm

# common parameters
pream_size = 50
model_tap_size = 2
batch_size = 200
est_path = 'model/tap2-est.bin'
eq_path = 'model/tap2-eq.bin'

# train parameters
data_size = 50000
train_snr = 20
train_size = 40000
est_fig = 'figure/tap2-est.png'
eq_fig = 'figure/tap2-eq.png'

# eval parameters
eval_size = 10000
payload_size = 200
eval_tap_size = 2
eval_snr = 100

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

    batches = eval_size // batch_size
    ber = 0
    for _ in tqdm(range(batches)):
        pream, pream_recv, payload_recv, label = offline.gen_ktap(batch_size, pream_size, eval_tap_size, eval_snr, payload_size)

        model.update_preamble(pream, pream_recv)
        payload_est = model.estimate(payload_recv)

        label_est = offline.demod_qpsk(payload_est)
        ber += offline.bit_error_rate(label_est, label, 2)

    return ber / batches

if __name__ == "__main__":
    parser = ArgumentParser('tap2')
    parser.add_argument('action', choices=('eval', 'train_est', 'train_eq'))
