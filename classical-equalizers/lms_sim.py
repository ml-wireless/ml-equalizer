import sys
sys.path.append('../')
import equalizer.util.offline as offline
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fftpack
from lms_eq import lms_model

# params for channel simulator
# @TODO change names to desriptions in comments (but everywhere else
#       in the code too)

model_tap_size = 2 # order of the linear, simulated channel
data_size = 1      # number of packets
train_snr = 25     # channel SNR

pream_size = 1000 #ideally ~40   # number of preamble symbols
payload_size = 2500 # number of payload symbols

# LMS parameters
mu = 0.01 # step size
order = 20 # num FIR taps

# convert QPSK symbols to 0,1,2,3 symbols per map:
#  1 | 3
# ---|---
#  0 | 2
def demod_qpsk(x):
    l = 2 * (x.real > 0) + 1 * (x.imag > 0)
    l = l.astype(np.int32)
    return l

if __name__ == "__main__":
    # gen received symbols using channel simulator
    # @TODO Why is tap a vector of tuples with a size corresponding to
    #       the number of symbols? Is the channel changing for each
    #       batch of symbols / packet?
    # @note pream is the true preamble, recv is the received preamble
    pream, pream_recv, payload_recv, tx_label = offline.gen_ktap(data_size,
            pream_size, model_tap_size, train_snr, payload_size, min_phase=True)

    print("pream:",pream.shape)
    print("pream_recv:",pream_recv.shape)

    print("label:",tx_label.shape)
    print("payload_recv:",payload_recv.shape)

    # alias received preamble symbols as x
    x = pream_recv[0]

    # alias desired preamble symbols as d
    d = pream[0]

    # convert x,d to numpy complex numbers
    x = x[:,0]+1j*x[:,1]
    d = d[:,0]+1j*d[:,1]

    lms = lms_model(order)
    # estimate inverse channel with LMS
    e = lms.inverse_channel(d,x,mu)
    print("Estimated inverse taps:",lms.get_inverse_channel())

    # Symbol equaliztion for payload
    x_payload = payload_recv[0]
    x_payload = x_payload[:,0]+1j*x_payload[:,1]
    est_payload = lms.estimate(x_payload)
    est_label = demod_qpsk(est_payload)
    rx_label = demod_qpsk(x_payload)

    # compute bit error rate (BER) as l1 norm of difference of actual
    # and estimated binary payload sequences
    ser_lms = payload_size - np.linalg.norm((est_label == tx_label)[0],1)
    ser_og = payload_size - np.linalg.norm((rx_label == tx_label)[0],1)

    print("BER (before)->(after) LMS: (",ser_og,")->(",ser_lms,")")

    ######################################
    # original plots for recieved and equalized preamble
    ######################################
    ndx = np.linspace(1,pream_size,pream_size)
    start = ndx.shape[0]-100

    plt.title("LMS Preamble Training Error")
    plt.plot(ndx,e.real,label="real")
    plt.plot(ndx,e.imag,label="imag")
    plt.legend()

    plt.figure(2)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(700,100,600,600)

    plt.title("Received & Equalized Payload Symbols")

    a = np.sqrt(2)/2.0
    label_map = {0: -1*a - 1j*a, 1: -1*a + 1j*a, 2: a - 1j*a, 3: a + 1j*a}
    label_to_sym = lambda t: label_map[t]
    tx_payload = np.array([label_to_sym(l) for l in tx_label[0]])

    plt.scatter(x_payload.real,x_payload.imag,label="received")
    plt.scatter(est_payload.real,est_payload.imag,label="equalized")
    plt.scatter(tx_payload.real,tx_payload.imag,label="transmitted")
    plt.legend()

    plt.show()
    ######################################
    # original procedure for recieved and equalized preamble
    ######################################
