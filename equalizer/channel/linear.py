import numpy as np
from numpy.lib.stride_tricks import as_strided

def real_tap(a):
    """
    return tap kernel from real numbers
    a: (*, k)
    returns: (*, k, 2, 2)
    """
    m = np.eye(2)
    for _ in range(2):
        a = np.expand_dims(a, -1)
    return m * a

def im_tap(a):
    """
    return tap kernel from complex numbers
    a: (*, k) dtype=np.complex_
    returns: (*, k, 2, 2)
    """
    # [[r, -i],
    #  [i,  r]]
    r, i = np.real(a), np.imag(a)
    col1 = np.stack((r, i), axis=-1)
    col2 = np.stack((-i, r), axis=-1)
    return np.stack((col1, col2), axis=-1)

def tap_proc(a, x):
    """
    process a k-tap channel
    `a`: tap kernel, (*, k, 2, 2)
    `x`: signal, (*, n, 2)
    returns: (*, n, 2)

    note that here tap kernel is 2x2 matrix
    use `real_tap` to convert real tap coefficients to matrix
    """
    # calculate the operand shape
    shapes = x.shape[:-1] + (a.shape[-3],) + x.shape[-1:]

    # note that this pad technique is the same as np.convolve(mode='same')
    width = a.shape[-3] - 1
    pads = ((0, 0), ) * (x.ndim - 2) + ((width - width // 2, width // 2), ) + ((0, 0), )
    x = np.pad(x, pads, mode='constant', constant_values=0)

    strides = x.strides[:-1] + (x.strides[-2], ) + x.strides[-1:]

    # stride hack + einsum
    x = as_strided(x, shapes, strides)
    return np.einsum('...nkj,...kij->...ni', x, np.flip(a, axis=-3), optimize='optimal')

def rot_mat(omega):
    """
    return a 2d rotation matrix
    `omega`: angle in rad, (*)
    `returns`: (*, 2, 2)
    """
    return im_tap(np.cos(omega) + np.sin(omega) * 1j)

def cfo_proc(omega, x):
    """
    process cfo
    `omega`: cfo rate, (*)
    `x`: signal, (*, n, 2)
    `returns`: (*, n, 2)
    """
    omega = np.expand_dims(omega, axis=-1) * np.arange(x.shape[-2], dtype=np.float)
    omega = rot_mat(omega)
    return np.einsum('...nj,...nij->...ni', x, omega, optimize='optimal')

def rms_power(x):
    """
    root mean square power of signal x
    `x`: (*, n, 2)
    `returns`: (*)
    """
    x = np.sum(x ** 2, axis=-1)
    return np.sqrt(np.mean(x, axis=-1))

def awgn_proc(snr, x):
    """
    awgn noise
    `snr`: (*)
    `x`: (*, n, 2)
    `returns`: (*, n, 2)
    """
    noise = rms_power(x) / (10 ** (snr / 20))
    shapes = noise.shape + x.shape[-2:]
    for _ in range(2):
        noise = np.expand_dims(noise, axis=-1)
    noise = noise * np.random.randn(*shapes)
    return x + noise

def inverse_tap_fft(a, expand, trunc, eps=0):
    """
    inverse a tap using FFT:
    a ->(DFT) A -> 1/(A+eps) ->(IDFT) b
    `a`: (*, k)
    `expand`: length to expand a, should be power of 2
    `trunc`: truncate to produce b
    `eps`: eps

    NOTE: assume real tap now
    """
    width = a.shape[-1]
    pad_left = expand // 2 - (width - 1) // 2
    pad_right = expand - width - pad_left
    a = np.pad(a, ((0, 0), ) * (a.ndim - 1) + ((pad_left, pad_right), ), mode='constant', constant_values=0)
    f = 1 / (np.fft.fft(a) + eps)
    a = np.real(np.fft.ifft(f))
    l = expand // 2 - (trunc - 1) // 2
    return a[..., l:l+trunc]

class LinearChannel(object):
    def __init__(self, tap_size, snr, max_cfo=None):
        """
        tap_size: self explained
        snr: self explained
        max_cfo: should in (0, pi) or None for no cfo, cfo will be generated in [-max_cfo, max_cfo]

        TODO: only real tap now
        """
        self.tap_size = tap_size
        self.snr = snr
        self.max_cfo = max_cfo
    
    def generateParameters(self, m=None):
        """
        m: batch size, None for no batch
        """
        shape = (self.tap_size,) if m == None else (m, self.tap_size)
        taps = np.random.randn(*shape)
        taps = taps / np.sqrt(np.sum(taps ** 2, axis=-1, keepdims=True))
        if self.max_cfo == None:
            return taps
        else:
            # output format TBD
            cfo = np.random.uniform(-self.max_cfo, self.max_cfo, m)
            return taps, cfo
    
    def process(self, param, x):
        if self.max_cfo == None:
            taps = param
        else:
            taps, cfo = param
        x = tap_proc(real_tap(taps), x)
        if self.max_cfo != None:
            x = cfo_proc(cfo, x)
        x = awgn_proc(self.snr, x)
        return x

"""
an example
"""
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    data = np.random.choice([-1, 1], size=(100, 2))
    label = data[:, 0] + 1 + (data[:, 1] + 1) // 2
    data = data / np.sqrt(2)

    def plot(ax, data, label, snr, cfo, tap):
        tap = np.array(tap)
        tap = tap / np.sqrt(np.sum(tap ** 2, axis=-1))
        data = tap_proc(real_tap(tap), data)
        data = cfo_proc(cfo, data)
        data = awgn_proc(snr, data)

        ax.set_title('snr {}\ncfo {}\ntap {}'.format(snr, cfo, tap))
        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='0.7')
        colors = ['y', 'g', 'r', 'b']
        for i in range(4):
            ax.scatter(data[label == i, 0], data[label == i, 1], color=colors[i])
    
    taps = [[0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0, 1]]
    snrs = [100, 20, 10]
    cfo = 0.0025
    _, axes = plt.subplots(4, 3, figsize=(18, 24))
    for i in range(4):
        for j in range(3):
            plot(axes[i, j], data, label, snrs[j], cfo, taps[i])
    plt.savefig('figure/channel.png')
