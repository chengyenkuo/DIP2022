import edge_detection as ed
import numpy as np
import basics
import masks
import math
import cv2

def Image2Cartesian(Rout, Cout):
    x, y = np.zeros((Rout,Cout)), np.zeros((Rout,Cout))
    for p in range(Rout):
        for q in range(Cout):
            x[p][q], y[p][q] = (q + 0.5), (Rout - p - 0.5)
    return x, y

def Cartesian2Image(x, y, Rin, Cin):
    (Rout, Cout), p, q = x.shape, np.zeros(x.shape), np.zeros(x.shape)
    
    for r in range(Rout):
        for c in range(Cout):
            p[r][c], q[r][c] = int(Rin-(int(y[r][c])+0.5)-0.5), int(x[r][c])
    return p.astype(int), q.astype(int)

def translation(x, y, Tx, Ty):
    return x - Tx, y - Ty

def scaling(x, y, Sx, Sy):
    return x / Sx, y / Sy

def rotation(x, y, theta):
    Xbar, Ybar = np.mean(x), np.mean(y)
    x, y = x - Xbar, y - Ybar
    x, y = (math.cos(-theta)*x - math.sin(-theta)*y), (math.sin(-theta)*x + math.cos(-theta)*y)
    return x + Xbar, y + Ybar

def mapping(source, p, q):
    (Rsource, Csource), (R, C), output = source.shape, p.shape, np.zeros(q.shape)
    
    for r in range(R):
        for c in range(C):
            if 0 <= p[r][c] and p[r][c] < Rsource and 0 <= q[r][c] and q[r][c] < Csource:
                output[r][c] = source[p[r][c]][q[r][c]]
    return output

def warping(source, K):
    (R,C), output, angle = source.shape, np.zeros(source.shape), 2*math.pi/180
    for i in range(R):
        for j in range(C):
            I, J = int(1.2*K*math.sin(j*angle)), int(K*math.cos(i*angle))
            if j + J < C and i + I < R:
                output[i][j] = source[(i+I)%R][(j+J)%C]
    return output

if __name__ == "__main__":
    pass