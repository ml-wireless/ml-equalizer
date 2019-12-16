from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import CNNEstimator
from tqdm import tqdm

pream_size = 5
model_tap_size = 2
batch_size = 200
est_path = 'model/cnn-est.bin'

data_size = 100000
train_snr = 10
train_size = 80000

if __name__ == "__main__":
    parser = ArgumentParser('train-cnn')
    parser.add_argument('epoch', nargs='?', type=int, default=10)
    args = parser.parse_args()

    pream, tap, recv = offline.gen_ktap(data_size, pream_size, model_tap_size, train_snr)
    model = CNNEstimator(model_tap_size)
    # model.load_state_dict(torch.load(est_path))
    # print(offline.batch_eval(model, (offline.to_torch(pream), offline.to_torch(recv)), offline.to_torch(tap), F.mse_loss, batch_size, silent=False))
    offline.train_e2e(model, (pream, recv), tap, F.mse_loss, train_size, batch_size, args.epoch, est_path, silent=False)
