import numpy as np
import math

def LoG_func(x, y, var):
    g = -(x*x + y*y) / (2*var)
    return -((1 + g) * math.exp(g) / (math.pi*var*var))
#
def LoG(window, var):
    mask = np.zeros((window, window))
    for i, y in zip(range(window), range(-(window//2), ((window+1)//2))):
        for j, x in zip(range(window), range(-(window//2), ((window+1)//2))):
            mask[i][j] = LoG_func(x, y, var)
    #print(np.sum(mask))
    return mask
#
def Gaussian_func(x, y, var):
    g = -(x*x + y*y) / (2*var)
    return math.exp(g) / (2*math.pi*var)
#
def Gaussian(window, var):
    mask = np.zeros((window, window))
    for i, y in zip(range(window), range(-(window//2), ((window+1)//2))):
        for j, x in zip(range(window), range(-(window//2), ((window+1)//2))):
            mask[i][j] = Gaussian_func(x, y, var)
    #print(np.sum(mask))
    return mask
#
def RG(K):
    return np.array([[-1, 0, 1], [-K, 0, K], [-1, 0, 1]]) / (K+2)
#
def CG(K):
    return np.array([[1, K, 1], [0, 0, 0], [-1, -K, -1]]) / (K+2)
#
def LP(K):
    return np.array([[1, K, 1], [K, K**2, K], [1, K, 1]]) / (K+2)**2