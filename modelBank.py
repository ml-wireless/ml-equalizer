import torch
from equalizer.model.classic import ZeroForcingEstimator, MMSEEstimator, LMSEstimator, FilterEqualizer, ClassicTap
from equalizer.model.tap import TapEstimator, CNNEstimator, HybridLmsEstimator

order = 31
order_lms = 5
expand = 8192
eps = 0
order_cnn = 5
path_cnn = 'model/cnn-est.bin'
path_cnn_zf = 'model/cnn-est-zf.bin'

zf_est = ZeroForcingEstimator(expand=expand, trunc=order, eps=eps)
mmse_est = MMSEEstimator(order, True)
cnn_est = CNNEstimator(order_cnn)
cnn_est.load_state_dict(torch.load(path_cnn))
hybrid_est = HybridLmsEstimator(cnn_est, order_cnn, 0.5)
lms_est = LMSEstimator(order_lms)
cnn_est_zf = CNNEstimator(order_cnn)
cnn_est_zf.load_state_dict(torch.load(path_cnn_zf))

model_bank = {
    "zf": ClassicTap(zf_est, FilterEqualizer),
    "mmse": ClassicTap(mmse_est, FilterEqualizer),
    "lms": ClassicTap(lms_est, FilterEqualizer),
    "cnn": ClassicTap(cnn_est, FilterEqualizer),
    "cnn-zf": ClassicTap(cnn_est_zf, FilterEqualizer),
    "hybrid": ClassicTap(hybrid_est, FilterEqualizer),
}
