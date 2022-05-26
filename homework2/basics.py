import numpy as np

def padding(source, window):
    (r, c), w = source.shape, window // 2
    L, R = np.flip(source[ :, 1:(w+1)], axis=1), np.flip(source[ :, (c-w):c], axis=1)
    output = np.hstack([L,source,R])
    U, D = np.flip(output[1:(w+1), :], axis=0), np.flip(output[(r-w):r, :], axis=0)
    output = np.vstack([U,output,D])
    return output