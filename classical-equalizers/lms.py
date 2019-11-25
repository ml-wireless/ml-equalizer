import sys
sys.path.append('../')
import equalizer.util.offline as offline
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fftpack

# LEFT OFF: Don't fully understand what the cost is and how to take its gradient. Also not sure how to apply a filter to IQ symbols (so tried converting them to amplitude samples). Current status is that it doesn't converge and is unstable / can explode to infinity for some intiial states

# params for channel simulator
# @TODO change names to desriptions in comments (but everywhere else
#       in the code too)
pream_size = 1000   # number of preamble symbols per packet
model_tap_size = 2 # order of the linear, simulated channel
data_size = 1      # number of packets
train_snr = 20     # channel SNR

# LMS parameters
mu = 0.01 # step size
order = 5 # num FIR taps

"""
Compute nth-step lms output

@param symbol complex sample of received signal
@param weights current lms FIR weights
@return nth-step lms output
"""
def lms(symbol, weights):
    return None

if __name__ == "__main__":
    # gen received symbols using channel simulator
    # @TODO Why is tap a vector of tuples with a size corresponding to
    #       the number of symbols? Is the channel changing for each
    #       batch of symbols / packet?
    # @note pream is the true preamble, recv is the received preamble
    pream, tap, recv = offline.gen_ktap(data_size,
            pream_size, model_tap_size, train_snr)

    print("pream:",pream.shape)
    print("pream:",pream)
    print("tap:",tap.shape)
    print("tap:",tap)
    print("recv:",recv.shape)
    print("recv:",recv)

    # alias received preamble symbols as x
    x = recv[0]

    # alias desired preamble symbols as d
    d = pream[0]

    # convert x,d to numpy complex numbers
    x = x[:,0]+1j*x[:,1]
    d = d[:,0]+1j*d[:,1]

    # convert complex symbols to real amplitude samples
    for i in range(0,x.shape[0]):
        x[i] = np.real(x[i]*np.exp(1j*i))
        d[i] = np.real(d[i]*np.exp(1j*i))

    # init weights to random real values
    w = np.array(np.random.normal(0,1,order)) + 1j*np.array(np.random.normal(0,1,order));

    # init empty array for equalized symbols
    y = np.zeros(pream_size)
    e = np.zeros(pream_size)
    ndx = np.linspace(1,pream_size,pream_size)

    # @note we start late since we need N-1 symbols of the past to
    # compute N tap FIR
    #
    # iterate through symbols in a single packet
    for i in range(order - 1, x.shape[0]):
        # apply the FIR to get current output
        y[i] = x[(i - (order - 1)):(i + 1)].T @ w
        # compute latest error
        e[i] = d[i] - y[i] # cost / error
        # update weights
        w = w + mu*e[i]*x[(i - (order - 1)):(i + 1)]
        #print("d[i]=",d[i],",y[i]=",y[i],",e[i]=",e,",w=",w)

    start = ndx.shape[0]-100

    plt.subplot(3,1,1).set_xlabel("symbol index")
    plt.title("Desired & Equalized Preamble")
    plt.plot(ndx[start:],d[start:],label="desired")
    plt.plot(ndx[start:],y[start:],label="equalized")
    plt.plot(ndx[start:],x[start:],label="received")
    plt.legend()

    #plt.subplot(5,1,3).set_xlabel("symbol index")
    #plt.title("Received Preamble")
    #plt.plot(ndx[start:],x[start:])

    plt.subplot(3,1,3).set_xlabel("symbol index")
    plt.title("LMS Error")
    plt.plot(ndx,e)
    plt.show()
