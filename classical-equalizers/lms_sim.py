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
order = 10 # num FIR taps

#temporary function to produce labels from complex symbols
#unsure about conversion
def temp_demod_qpsk(x):
    l = 2 * (x[...].real > 0) + 1 * (x[...].imag > 0)
    l = l.astype(np.int32)
    return l

if __name__ == "__main__":
    # gen received symbols using channel simulator
    # @TODO Why is tap a vector of tuples with a size corresponding to
    #       the number of symbols? Is the channel changing for each
    #       batch of symbols / packet?
    # @note pream is the true preamble, recv is the received preamble
    pream, pream_recv, payload_recv, label = offline.gen_ktap(data_size,
            pream_size, model_tap_size, train_snr, payload_size)



    print("pream:",pream.shape)
    print("pream:",pream)
    print("pream_recv:",pream_recv.shape)
    print("pream_recv:",pream_recv)

    print("label:",label.shape)
    print("label:",label)
    print("payload_recv:",payload_recv.shape)
    print("payload_recv:",payload_recv)


    # alias received preamble symbols as x
    x = pream_recv[0]

    # alias desired preamble symbols as d
    d = pream[0]

    # convert x,d to numpy complex numbers
    x = x[:,0]+1j*x[:,1]
    d = d[:,0]+1j*d[:,1]

    lms = lms_model(order)
    #To eval performance over preamble, generate large preamble and use smaller slice to calculate channel inverse
    lms.inverse_channel(d,x,mu=mu)
    y = lms.estimate(x)
    e = d-y

    # Symbol equaliztion for payload
    x_payload = payload_recv[0]
    x_payload = x_payload[:,0]+1j*x_payload[:,1]
    est_payload = lms.estimate(x_payload)
    est_label = temp_demod_qpsk(est_payload[:2])



######################################
# original plots for recieved and equalized preamble
######################################
    ndx = np.linspace(1,pream_size,pream_size)
    start = ndx.shape[0]-100

    plt.subplot(3,1,1).set_xlabel("symbol index")
    plt.title("Desired & Equalized Preamble")
    plt.plot(ndx[start:],d[start:],label="desired")
    plt.plot(ndx[start:],y[start:],label="equalized")
    plt.plot(ndx[start:],x[start:],label="received")
    plt.legend()


    plt.subplot(3,1,3).set_xlabel("symbol index")
    plt.title("LMS Error")
    plt.plot(ndx,e.real,label="real")
    plt.plot(ndx,e.imag,label="imag")
    plt.legend()


    plt.figure(2)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(700,100,600,600)

    plt.title("Received & equalized symbols")
    plt.scatter(x_payload[start:].real,x_payload[start:].imag,label="received")
    plt.scatter(est_payload[start:].real,est_payload[start:].imag,label="equalized")
    plt.legend()

    plt.show()
######################################
# original procedure for recieved and equalized preamble
######################################
