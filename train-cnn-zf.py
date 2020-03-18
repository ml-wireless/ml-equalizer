import gzip, pickle
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import CNNEstimator
from tqdm import tqdm

dataPath = 'data/zf_data_snr10_pream40.pkl'
model_tap_size = 5
batch_size = 200
est_path = 'model/cnn-est.bin'
train_size = 8000

def load_data(path):
    with open(dataPath, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1], data[2]

def pack_weight(tap):
    return np.concatenate((np.real(tap), np.imag(tap)), axis=-1)

if __name__ == "__main__":
    parser = ArgumentParser('train-cnn-zf')
    parser.add_argument('epoch', nargs='?', type=int, default=10)
    args = parser.parse_args()
    
    pream, pream_recv, inverse = load_data(dataPath)
    inverse = pack_weight(inverse)

    model = CNNEstimator(model_tap_size)
    offline.train_e2e(model, (pream, pream_recv), inverse, F.mse_loss, train_size, batch_size, args.epoch, est_path, silent=False)