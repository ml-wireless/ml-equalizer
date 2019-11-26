import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fftpack

plt.figure(2) # channel characteristics on bottom
plt.figure(3) # spectrum second
plt.figure(1) # waveform and LMS error on top

# params
Tb = 1 # time per bit (sec)
fb = 4 # samples per bit
Ts = Tb/fb # sampling period
fs = fb/Tb # sampling frequency
fc = .1 # carrier frequency
preamble_size = 60 # determines point after which MSE is computed
preamble_samples = preamble_size*fb

#qpsk with Gray Coding
qpsk = np.array([[3,2],[4,1]])

assert(fc <= (fs/2)) # nyquist

######################################################################
# Create transmit waveform
######################################################################

# create dummy binary data
x = np.array([0,0,1,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,1,0,
    0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,0,0,
    0,0,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,0,0,0,1,
    1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,1,1,1,1,0,
    0,0,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,0,0,0,1,
    0,1,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,
    0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,0,0,
    1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,1,0,0,
    0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,1,0])

assert(preamble_size < x.shape[0])

#temporary padding
if len(x)%2==1:
    x = np.append(x, 0)

# create time sequence
t_bits = np.linspace(1,x.shape[0]*fb,x.shape[0]*fb)*Ts

# upsample binary sequence
xu_vector = np.array([np.ones(int(1/Ts))*i for i in x])
# entries in xu_vector are vectors of N of the same bit values from x
xu = xu_vector.flatten()

plt.subplot(9,1,1).set_xlabel("t")
plt.plot(t_bits,xu)
plt.title("Binary sequence")

## encode with NRZ (0 -> -1, 1 -> 1)
#xe = np.array([i - int(i==0) for i in xu])
qpsk_phase = 1 + x[::2] + 2**x[1::2]
t = np.linspace(1,qpsk_phase.shape[0]*fb,qpsk_phase.shape[0]*fb)*Ts
up_qpsk_phase = np.array([np.ones(int(1/Ts))*i for i in qpsk_phase]).flatten()
xm = np.cos(2*np.pi*fc*t + np.pi*(2*up_qpsk_phase - 1)/4)

#xm = np.cos(np.pi*t)
plt.subplot(9,1,3).set_xlabel("t")
plt.plot(t,xm)
plt.title("Modulated Waveform")


######################################################################
# Create and visualize the channel
######################################################################
ct = [-.25, -.1] # channel taps

channel = signal.dlti([ct[0], ct[1]], [1, 0], dt=Ts)
t2, channel_imp = channel.impulse()
w, mag, phase = channel.bode()

# plot impulse, frequency, and phase response
plt.figure(2)
plt.subplot(5,1,1)
plt.title("Channel Impulse Response")
plt.plot(t2,np.squeeze(channel_imp))
plt.subplot(5,1,3)
plt.title("Magnitude Response")
plt.semilogx(w, mag)
plt.subplot(5,1,5)
plt.title("Phase Response")
plt.semilogx(w, phase)

######################################################################
# Convolve with channel, add noise
######################################################################
y_clean = np.zeros(xm.shape[0])

for i in range(1, xm.shape[0]):
    y_clean[i] = xm[i]*ct[0] + xm[i-1]*ct[1]

mu = 0
sigma = 1
# @TODO parameterize awgn in terms of SNR
# @TODO put awgn back into channel output once LMS works w/o
y = y_clean + .08*np.random.normal(mu, sigma, y_clean.shape[0])

mse_y = ((xm[10:] - y[10:])**2).mean()
mse_y_str = str(round(mse_y,5))

plt.figure(1)
plt.subplot(9,1,5).set_xlabel("t")
plt.plot(t,y)
plt.title("Received Waveform (channel + awgn) (MSE=" + mse_y_str + ")")

######################################################################
# Test perturbing estimation of channel
######################################################################
ct = ct + np.array([.00, .00])

######################################################################
# Test ZFE with a-priori known channel
######################################################################
yh_zfe = np.zeros(y.shape[0])

for i in range(1, y.shape[0]):
    yh_zfe[i] = (1/ct[0])*y[i] + (-1*ct[1]/ct[0])*yh_zfe[i-1]

######################################################################
# Test LMS with learned channel
######################################################################
yh_lms = np.zeros(y.shape[0])

mu = .095 # step size
N = 5    # order
w = np.zeros(N) # N rand weights
d = xm # alias for readability
e = np.zeros(y.shape[0])

for i in range(N-1, y.shape[0]):
    # apply LMS FIR to get current output sample
    yh_lms[i] = y[(i-(N-1)):(i+1)].T @ w
    # compute latest error
    e[i] = d[i] - yh_lms[i]
    # only run LMS over preamble, stop learning after preamble
    if (i-(N-1) < preamble_samples):
        # update weights accordingly
        w = w + mu*e[i]*y[(i-(N-1)):(i+1)]

######################################################################
# Compute MSE between transmitted waveform and equalized waveforms
######################################################################
# @note offset the sequences so mean isn't computed based on edge
#       conditions (e.g. filter time delay)
mse = {} # og, zfe, lms, zfe_i, lms_i
dec = 4
start = preamble_samples # start sample
mse['og'] = ((xm[start:] - y[start:])**2).mean()
mse['zfe'] = ((xm[start:] - yh_zfe[start:])**2).mean()
mse['lms'] = ((xm[start:] - yh_lms[start:])**2).mean()
mse['og_str'] = str(round(mse['og'],dec))
mse['zfe_str'] = str(round(mse['zfe'],dec))
mse['lms_str'] = str(round(mse['lms'],dec))
mse['zfe_i_str'] = str(round(mse['og']/mse['zfe'],dec))
mse['lms_i_str'] = str(round(mse['og']/mse['lms'],dec))

print("Mean Square Error (OG) : " + mse['og_str'])
print("Mean Square Error (ZFE): " + mse['zfe_str'])
print("----- improved by -----> " + mse['zfe_i_str'])
print("Mean Square Error (LMS): " + mse['lms_str'])
print("----- improved by -----> " + mse['lms_i_str'])

plt.figure(1)

plt.subplot(9,1,7).set_xlabel("t")
plt.title("Equalized waveforms (MSE(OG)="+mse['og_str']+")")
plt.plot(t,yh_zfe,label="ZFE")
plt.plot(t,yh_lms,label="LMS")
plt.legend()

mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(100,100,1200, 900)

# show the LMS and ZFE error here too
plt.subplot(9,1,9).set_xlabel("t")
plt.title("ZFE (MSE="+mse['zfe_str']+" i-by=" + mse['zfe_i_str'] + "x) and LMS (MSE="+mse['lms_str']+" i-by=" + mse['lms_i_str'] + "x) error")
plt.plot(t,d-yh_zfe, label="ZFE") # ZFE
plt.plot(t,e, label="LMS (PreT=" + str(preamble_samples*Ts) + ", PreSym=" + str(preamble_size) + ")") # LMS
plt.legend()


######################################################################
# Plot spectrum of transmitted signal, received signal, and equalized
# signal
######################################################################
plt.figure(3)
plt.title("Spectrum of Tx, Rx, and equalized(Rx)")

N = xm.shape[0]
xf = np.linspace(0.0, 1.0/(2.0*Ts), N/2)

# transmitted signal
yf = fftpack.fft(xm)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label="Tx")

# received signal
yf = fftpack.fft(y)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label="Rx (MSE1="+mse_y_str+")")

# equalized signal (ZFE)
yf = fftpack.fft(yh_zfe)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label="ZFE(Rx): MSE(ZFE)="+mse['zfe_str'])

# equalized signal (ZFE)
yf = fftpack.fft(yh_lms)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label="LMS(Rx): MSE(LMS)="+mse['lms_str'])
plt.legend()

plt.show()
