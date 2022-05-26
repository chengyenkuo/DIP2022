import numpy as np
import cv2

def g2b(source):
    return (source//255)
def b2g(source):
    return (source*255)
def NOT(source):  # complement
    return ((source + np.ones(source.shape)) % 2).astype(int)
def OR(sourceA, sourceB):   # union
    (R, C), out=sourceA.shape, np.zeros(sourceA.shape)
    for i in range(R):
        for j in range(C):
            if sourceA[i][j] or sourceB[i][j]:
                out[i][j] = 1
    return out.astype(int)
def AND(sourceA, sourceB):  # intersection
    (R, C), out=sourceA.shape, np.zeros(sourceA.shape)
    for i in range(R):
        for j in range(C):
            if sourceA[i][j] and sourceB[i][j]:
                out[i][j] = 1
    return out.astype(int)
def XOR(sourceA, sourceB):  # exclusive-OR
    (R, C), out=sourceA.shape, np.zeros(sourceA.shape)
    for i in range(R):
        for j in range(C):
            if (not sourceA[i][j] and sourceB[i][j]) or (sourceA[i][j] and not sourceB[i][j]):
                out[i][j] = 1
    return out.astype(int)
def R(source):  # reflection
    R, C = source.shape
    mR, mC, output = (R//2), (C//2), np.copy(source)
    for r in range(R//2):
        for c in range(C):
            dR, dC = (mR-r), (mC-c)
            output[r][c], output[mR+dR-1][mC+dC-1] = source[mR+dR-1][mC+dC-1], source[r][c]
    return output.astype(int)
def padding(source, window):
    (r, c), w = source.shape, window // 2
    L, R = np.flip(source[ :, 1:(w+1)], axis=1), np.flip(source[ :, (c-w):c], axis=1)
    out = np.hstack([L,source,R])
    U, D = np.flip(out[1:(w+1), :], axis=0), np.flip(out[(r-w):r, :], axis=0)
    out = np.vstack([U,out,D])
    return out
if __name__ == '__main__':
    sourceA = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ])
    sourceB = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0]
    ])
    print(R(sourceB))


#https://towardsdatascience.com/implementing-a-connected-component-labeling-algorithm-from-scratch-94e1636554f
#https://dl.acm.org/doi/pdf/10.1145/357994.358023