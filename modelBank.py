import torch
from equalizer.model.classic import ZeroForcingEstimator, MMSEEstimator, LMSEstimator, FilterEqualizer, ClassicTap, DummyTap
from equalizer.model.tap import TapEstimator, CNNEstimator, HybridLmsEstimator

path_cnn = 'model/cnn-est.bin'
path_cnn_31 = 'model/cnn-est-31.bin'
path_cnn_zf = 'model/cnn-est-zf.bin'

def load_est_model(order, path):
    ret = CNNEstimator(order)
    ret.load_state_dict(torch.load(path))
    return ret

cnn_est = load_est_model(5, path_cnn)
cnn_est_31 = load_est_model(31, path_cnn_31)
hybrid_est = HybridLmsEstimator(cnn_est, mu=0.03, split=0.5)
hybrid_est_31 = HybridLmsEstimator(cnn_est_31, mu=0.01, split=1)

est_bank = {
    "zf-31": ZeroForcingEstimator(expand=8192, trunc=31, eps=0),
    "zf-5": ZeroForcingEstimator(expand=8192, trunc=5, eps=0),
    "mmse-31": MMSEEstimator(31, True),
    "mmse-5": MMSEEstimator(5, True),
    "lms-5": LMSEstimator(5, mu=0.1),
    "lms-31": LMSEstimator(31, mu=0.04),
    "cnn-5": cnn_est,
    "cnn-zf": load_est_model(31, path_cnn_zf),
    "cnn-31": cnn_est_31,
    "hybrid-5": hybrid_est,
    "hybrid-31": hybrid_est_31,
}

model_bank = { k: ClassicTap(v, FilterEqualizer) for k, v in est_bank.items() }
model_bank['dummy'] = DummyTap()
