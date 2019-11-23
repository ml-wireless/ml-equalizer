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
print("x:",x)

# create time sequence
t = np.linspace(1,x.shape[0]*fb,x.shape[0]*fb)*Ts
print("t:",t)

# upsample binary sequence
xu_vector = np.array([np.ones(int(1/Ts))*i for i in x])
# entries in xu_vector are vectors of N of the same bit values from x
xu = xu_vector.flatten()
print("xu:",xu)

plt.subplot(7,1,1)
plt.plot(t,xu)
plt.title("Binary sequence")

## encode with NRZ (0 -> -1, 1 -> 1)
#xe = np.array([i - int(i==0) for i in xu])
#print("xe:",xe)

# modulate with BPSK
xm = np.cos(2*np.pi*fc*t + np.pi*xu)
#xm = np.cos(np.pi*t)
print("xm:",xm)

plt.subplot(7,1,3)
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
y = y_clean + .1*np.random.normal(mu, sigma, y_clean.shape[0])

plt.figure(1)
plt.subplot(7,1,5)
plt.plot(t,y)
plt.title("Received Waveform (channel + awgn)")

######################################################################
# Test perturbing estimation of channel
######################################################################
ct = ct + np.array([.00, .00])

######################################################################
# Test ZFE with a-priori known channel
######################################################################
yh = np.zeros(y.shape[0])

for i in range(1, y.shape[0]):
    yh[i] = (1/ct[0])*y[i] + (-1*ct[1]/ct[0])*yh[i-1]


######################################################################
# Compute MSE between transmitted waveform and equalized waveform
######################################################################
# @note offset the sequences so mean isn't computed based on edge
#       conditions (e.g. filter time delay)
mse = ((xm[10:] - yh[10:])**2).mean()
mse_str = str(round(mse,5))
print("Mean Square Error: ",mse)

plt.subplot(7,1,7)

plt.title("Equalized waveform (channel^-1) (MSE="+mse_str+")")
plt.plot(t,yh)

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
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label="Rx")

# equalized signal
yf = fftpack.fft(yh)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label="EQ(Rx) (MSE="+mse_str+")")
plt.legend()

plt.show()
