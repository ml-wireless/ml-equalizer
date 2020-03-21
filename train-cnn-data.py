import gzip, pickle
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import CNNEstimator
from tqdm import tqdm

# dataPath = 'data/lms_all_snr_10.gz'
model_tap_size = 5
batch_size = 200
# est_path = 'model/cnn-est.bin'
train_size = 16000

def load_data(path):
    with gzip.open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data['pream'], data['pream_recv'], data['inverse_weights'], data['gen_taps']

def pack_weight(tap):
    return np.concatenate((np.real(tap), np.imag(tap)), axis=-1)

if __name__ == "__main__":
    parser = ArgumentParser('train-cnn-data')
    parser.add_argument('-d', '--data')
    parser.add_argument('-o', '--output')
    parser.add_argument('epoch', nargs='?', type=int, default=10)
    args = parser.parse_args()

    pream, pream_recv, inverse, tap = load_data(args.data)
    inverse = pack_weight(inverse)

    model = CNNEstimator(model_tap_size)
    offline.train_e2e(model, (pream, pream_recv), inverse, F.mse_loss, train_size, batch_size, args.epoch, args.output, silent=False)
