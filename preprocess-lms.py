import gzip, pickle
from glob import glob
import numpy as np

dataPath = glob('data/raw/lms_data_snr_10_*.gz')
output = 'data/lms_all_snr_10.gz'
cols = ['pream', 'pream_recv', 'inverse_weights', 'gen_taps']

def load_data(path):
    with gzip.open(path, 'rb') as fp:
        data = pickle.load(fp)
    ret = {}
    for col in cols:
        ret[col] = np.stack(data[col])
    return ret

def combine_data(*rets):
    ret = {}
    for col in cols:
        ret[col] = np.concatenate(list(map(lambda x: x[col], rets)), axis=0)
    return ret

if __name__ == "__main__":
    data_all = combine_data(*map(load_data, dataPath))
    with gzip.open(output, 'wb') as fp:
        pickle.dump(data_all, fp)
