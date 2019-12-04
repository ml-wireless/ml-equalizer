import numpy as np
import util.offline as offline

# mmse with unknown channel
def mmse1(recv, pream, order):

    temp_pream = (pream[:, :, 0] + pream[:, :, 1]*1j).reshape(recv.shape[1],1)
    temp_recv = (recv[:, :, 0] + recv[:, :, 1]*1j).reshape(recv.shape[1],1)
    temp_recv = np.pad(temp_recv, ((order,order), (0,0)),'constant', constant_values=(0j,0j))

    R = np.zeros((2*order+1, 2*order+1),dtype=np.complex_)
    d = np.zeros((2*order+1, 1),dtype=np.complex_)
    for i in range (order, recv.shape[1]+order) :        
        R += np.matmul( temp_recv[i-order: i+order+1], temp_recv[i-order: i+order+1].conj().T)
        d += temp_pream[i-order] * temp_recv[i-order: i+order+1]
    R = R / len(recv)
    d = d / len(recv)

    return np.linalg.solve(R, d) , temp_recv, temp_pream

# mmse with known channel
def mmse2(recv, tap):
    temp_recv = (recv[:, :, 0] + recv[:, :, 1]*1j)
    H = np.eye(recv.shape[0]) * tap[0][0]
    for i in range(1,recv.shape[0]):
        H[i][i-1] = tap[0][1]
    return np.linalg.solve((H @ H.T), H.T @ temp_recv)


pream, tap, recv = offline.gen_ktap(1, 5, 2, 10)
G , temp_recv, temp_pream = mmse1(recv, pream, 3)

pream_hat = np.convolve((recv[:, :, 0] + recv[:, :, 1]*1j).flatten(), G.flatten() , mode="full")
print(pream_hat)
print(temp_pream)

x, tap, y = offline.gen_ktap(10, 5, 2, 10)
x_hat = mmse2(y, tap)