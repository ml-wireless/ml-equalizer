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
    # [[cos(x), -sin(x)],
    #  [sin(x),  cos(x)]]
    co = np.cos(omega)
    si = np.sin(omega)
    col1 = np.stack((co, si), axis=-1)
    col2 = np.stack((-si, co), axis=-1)
    return np.stack((col1, col2), axis=-1)

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

def awgn_proc(x, snr):
    """
    awgn noise
    `x`: (*, n, 2)
    `snr`: (*)
    `returns`: (*, n, 2)
    """
    noise = rms_power(x) / (10 ** (snr / 20))
    shapes = noise.shape + x.shape[-2:]
    for _ in range(2):
        noise = np.expand_dims(noise, axis=-1)
    noise = noise * np.random.randn(*shapes)
    return x + noise

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
        data = awgn_proc(data, snr)

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
