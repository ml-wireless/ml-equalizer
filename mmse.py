import numpy as np
import equalizer.util.offline as offline
from equalizer.channel.linear import im_tap, tap_proc

# mmse with known channel
def mmse2(recv, tap):
    temp_recv = (recv[:, :, 0] + recv[:, :, 1]*1j)
    H = np.eye(recv.shape[0]) * tap[0][0]
    for i in range(1,recv.shape[0]):
        H[i][i-1] = tap[0][1]
    return np.linalg.solve((H @ H.T), H.T @ temp_recv)

# pream, tap, recv = offline.gen_ktap(2, 10, 2, 100)
# G, temp_recv, temp_pream = mmse1(recv, pream, 3)

# inv_tap = im_tap(G)
# pream_hat = tap_proc(inv_tap, recv)
# print(pream_hat - pream)

# pream_hat = np.convolve(G.flatten(), (recv[:, :, 0] + recv[:, :, 1]*1j).flatten(), mode="same")

# x, tap, y = offline.gen_ktap(10, 5, 2, 10)
# x_hat = mmse2(y, tap)
