import numpy as np
import equalizer.util.offline as offline
from equalizer.channel.linear import im_tap, tap_proc

# mmse with unknown channel
def mmse1(recv, pream, order):
    temp_pream = pream[..., 0] + pream[..., 1] * 1j
    temp_recv = recv[..., 0] + recv[..., 1] * 1j
    temp_recv = np.pad(temp_recv, ((0, 0),) * (temp_recv.ndim - 1) + ((order, order),), 'constant')

    R = np.zeros(temp_recv.shape[:-1] + (2 * order + 1, 2 * order + 1), dtype=np.complex_)
    d = np.zeros(temp_recv.shape[:-1] + (2 * order + 1,), dtype=np.complex_)
    for i in range(order, recv.shape[1]+order):
        Ri = temp_recv[..., i-order:i+order+1]
        R += np.expand_dims(Ri, -2) * np.expand_dims(Ri, -1)
        d += temp_pream[..., i-order:i-order+1] * Ri
    return np.flip(np.linalg.solve(R, d), -1), temp_recv, temp_pream

# mmse with known channel
def mmse2(recv, tap):
    temp_recv = (recv[:, :, 0] + recv[:, :, 1]*1j)
    H = np.eye(recv.shape[0]) * tap[0][0]
    for i in range(1,recv.shape[0]):
        H[i][i-1] = tap[0][1]
    return np.linalg.solve((H @ H.T), H.T @ temp_recv)

pream, tap, recv = offline.gen_ktap(2, 10, 2, 100)
G, temp_recv, temp_pream = mmse1(recv, pream, 3)

inv_tap = im_tap(G)
pream_hat = tap_proc(inv_tap, recv)
print(pream_hat - pream)

# pream_hat = np.convolve(G.flatten(), (recv[:, :, 0] + recv[:, :, 1]*1j).flatten(), mode="same")

# x, tap, y = offline.gen_ktap(10, 5, 2, 10)
# x_hat = mmse2(y, tap)
