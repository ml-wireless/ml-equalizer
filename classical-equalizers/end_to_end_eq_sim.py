import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fftpack

plt.figure(1)

# params
Tb = 1 # time per bit (sec)
fb = 4 # samples per bit
Ts = Tb/fb # sampling period
fs = fb/Tb # sampling frequency
fc = .1 # carrier frequency

assert(fc <= (fs/2)) # nyquist

######################################################################
# Create transmit waveform
######################################################################

# create dummy binary data
x = np.array([0,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,1,0,
    0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,0,0,
    0,0,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,0,0,0,1,
    1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,1,1,1,1,0,
    0,1,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0])

# create time sequence
t = np.linspace(1,x.shape[0]*fb,x.shape[0]*fb)*Ts

# upsample binary sequence
xu_vector = np.array([np.ones(int(1/Ts))*i for i in x])
# entries in xu_vector are vectors of N of the same bit values from x
xu = xu_vector.flatten()

plt.subplot(9,1,1)
plt.plot(t,xu)
plt.title("Binary sequence")

## encode with NRZ (0 -> -1, 1 -> 1)
#xe = np.array([i - int(i==0) for i in xu])

# modulate with BPSK
xm = np.cos(2*np.pi*fc*t + np.pi*xu)
#xm = np.cos(np.pi*t)

plt.subplot(9,1,3)
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
y = y_clean# + .1*np.random.normal(mu, sigma, y_clean.shape[0])

mse_y = ((xm[10:] - y[10:])**2).mean()
mse_y_str = str(round(mse_y,5))

plt.figure(1)
plt.subplot(9,1,5)
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
# @TODO Fix the cost function for updating weights
######################################################################
yh_lms = np.zeros(y.shape[0])

mu = 0.001 # step size
N = 15      # order
#w = np.array(.1*np.ones(N)) # .1 is an arbitrary non-zero value
#w = np.random.normal(0,1,N) # N rand weights
w = np.zeros(N) # N rand weights
d = xm # alias for readability
e = np.zeros(y.shape[0])

for i in range(N-1, y.shape[0]):
    # apply FIR to get current output sample
    yh_lms[i] = y[(i-(N-1)):(i+1)].T @ w
    # compute latest error
    e[i] = d[i] - yh_lms[i]
    # update weights accordingly
    w = w + mu*e[i]*y[(i-(N-1)):(i+1)]

######################################################################
# Compute MSE between transmitted waveform and equalized waveforms
######################################################################
# @note offset the sequences so mean isn't computed based on edge
#       conditions (e.g. filter time delay)
mse = {} # og, zfe, lms, zfe_i, lms_i
dec = 4
mse['og'] = ((xm[10:] - y[10:])**2).mean()
mse['zfe'] = ((xm[10:] - yh_zfe[10:])**2).mean()
mse['lms'] = ((xm[10:] - yh_lms[10:])**2).mean()
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

plt.subplot(9,1,7)
plt.title("Equalized waveforms (MSE(OG)="+mse['og_str']+"_)")
plt.plot(t,yh_zfe,label="ZFE (MSE="+mse['zfe_str']+")")
plt.plot(t,yh_lms,label="LMS (MSE="+mse['lms_str']+")")
plt.legend()

mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(1000,100,640, 545)

# show the LMS error here too
plt.subplot(9,1,9)
plt.title("LMS error")
plt.plot(t,e)

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

######################################################################
# Plot LMS performance
######################################################################

plt.show()
