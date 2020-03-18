from argparse import ArgumentParser
import torch
import numpy as np
import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import CNNEstimator
from tqdm import tqdm

# pream_size = 40
model_tap_size = 2
batch_size = 200
est_path = 'model/cnn-est.bin'

# eval parameters
eval_size = 10000
pream_test_sizes = [32, 40, 60, 80, 100, 120, 128]
eval_snrs = [1, 10, 50, 100, 500, 1000, 5000]
eval_dat = 'figure/cnn-preLen.dat'

if __name__ == "__main__":
    parser = ArgumentParser('test-cnn')
    parser.add_argument('epoch', nargs='?', type=int, default=10)
    args = parser.parse_args()

    model = CNNEstimator(model_tap_size)
    model.load_state_dict(torch.load(est_path))

    #test
    with open(eval_dat, 'w') as f:
        for pream_test_size in pream_test_sizes:
            test_loss = [pream_test_size]

            for snr in eval_snrs:
                pream_test, tap_test, recv_test = offline.gen_ktap(batch_size, pream_test_size, model_tap_size, snr)

                test_loss_epoch = []
                for i in range(args.epoch):
                    print("test on epoch {}".format(i + 1))
                    test_loss_epoch.append(offline.batch_eval(model, (offline.to_torch(pream_test), offline.to_torch(recv_test)), offline.to_torch(tap_test), F.mse_loss, batch_size))     
                    print("epoch {}  test loss: {}".format(i + 1, test_loss_epoch[-1]))

                test_loss.append(np.mean(test_loss_epoch))
        
            for i in range(len(test_loss)):
                print(' ' + str(test_loss[i]), file = f, end = '')
            print('', file = f)