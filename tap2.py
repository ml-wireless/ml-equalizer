import torch.nn.functional as F
import equalizer.util.offline as offline
from equalizer.model.tap import TapEstimator, TapEqualizer

data_size = 50000
seq_size = 50
tap_size = 2
snr = 20
train_size = 40000
batch_size = 200
epoch = 50

estimator = TapEstimator(seq_size, tap_size)
equalizer = TapEqualizer(tap_size)

gen_data = lambda :offline.gen_ktap(data_size, seq_size, tap_size, snr)

# train estimator
# send, label, param, recv = gen_data()
# offline.train_e2e(estimator, (send, recv), param, F.mse_loss, train_size, batch_size, epoch, 'model/tap2-est.bin')

# train equalizer
send, label, param, recv = gen_data()
offline.train_e2e(equalizer, (param, recv), send, F.mse_loss, train_size, batch_size, epoch, 'model/tap2-eq.bin', silent=False)
