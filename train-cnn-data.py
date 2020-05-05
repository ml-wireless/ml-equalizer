import gzip, pickle
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import CNNEstimator
from tqdm import tqdm

batch_size = 200
train_split = 0.8
# train_size = 16000

def load_data(path):
    with gzip.open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data['pream'], data['pream_recv'], data['inverse_weights'], data['gen_taps']

def pack_weight(tap):
    return np.concatenate((np.real(tap), np.imag(tap)), axis=-1)

if __name__ == "__main__":
    parser = ArgumentParser('train-cnn-data')
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-f', '--figure', default="")
    parser.add_argument('epoch', nargs='?', type=int, default=10)
    args = parser.parse_args()

    pream, pream_recv, inverse, tap = load_data(args.data)
    model = CNNEstimator(inverse.shape[-1])
    inverse = pack_weight(inverse)
    train_size = int(np.floor(train_split * pream.shape[0]))
    train_loss, test_loss = offline.train_e2e(model, (pream, pream_recv), inverse, F.mse_loss, train_size, batch_size, args.epoch, args.output, silent=False)
    if args.figure != "":
        with open(args.figure, 'a') as f:
            for i in range(train_loss.shape[0]):
                print(train_loss[i], test_loss[i], file=f)
