import numpy as np
import math
import cv2
import sys

def extraction(source):
    R, C, D = source.shape
    output = np.zeros((R//2, C//2, D))
    
    for r in range(R//2):
        for c in range(C//2):
            for d in range(D):
                for i in range(2):
                    for j in range(2):
                        output[r][c][d] += source[2*r+i][2*c+j][d]
    return output

def error_diffussion(source, r, c, d):
    output = np.pad(source, ((1, 1), (1, 1), (0, 0)), mode='constant')
    output[r+1][c+1][d] -= 1
    R, C, D = source.shape
    
    if r == 0 and c == 0:
        mask = np.array([[0, 0, 0], [0, -5, 2], [0, 2, 1]])/5.
    elif r == 0 and c == (C-1):
        mask = np.array([[0, 0, 0], [2, -5, 0], [1, 2, 0]])/5.
    elif r == (R-1) and c == 0:
        mask = np.array([[0, 2, 1], [0, -5, 2], [0, 0, 0]])/5.
    elif r == (R-1) and c == (C-1):
        mask = np.array([[1, 2, 0], [2, -5, 0], [0, 0, 0]])/5.
    elif r == 0:
        mask = np.array([[0, 0, 0], [2, -8, 2], [1, 2, 1]])/8.
    elif r == (R-1):
        mask = np.array([[1, 2, 1], [2, -8, 2], [0, 0, 0]])/8.
    elif c == 0:
        mask = np.array([[0, 2, 1], [0, -8, 2], [0, 2, 1]])/8.
    elif c == (C-1):
        mask = np.array([[1, 2, 0], [2, -8, 0], [1, 2, 0]])/8.
    else:
        mask = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]])/12.
    #print(mask)
    val = output[r+1][c+1][d]
    for i in range(3):
        for j in range(3):
            output[r+i][c+j][d] += val*mask[i][j]
    return output[1:-1, 1:-1, :]

sample = cv2.imread(sys.argv[1])
R, C, D = sample.shape
X, B, L = [sample/255.], [np.zeros((R, C, D))], int(math.log(R, 2)) + 1
E = [X[0]-B[0]]

for i in range(1, L):
    X = [extraction(X[0])] + X
    B = [np.zeros((R//(2**i), C//(2**i), D))] + B
    E = [X[0] - B[0]] + E

print(E[0][:, :, 0])
print(E[1][:, :, 0])
print(E[2][:, :, 0])
print(E[3][:, :, 0])

for d in range(D):
    print(f"Dimension {d}...")
    for k in range(int(X[0][:, :, d])):
        print(k)
        r, c = 0, 0
        for l in range(1, L):
            r, c, = r*2, c*2
            dr, dc = divmod(np.argmax(E[l][r:r+2, c:c+2, d]), 2)
            r, c = r+dr, c+dc
        print(r, c)
        B[L-1][r, c, d] = 1
        E = [error_diffussion(E[L-1], r, c, d)]
        for i in range(1, L):
            E = [extraction(E[0])] + E
        if k % 1000 == 0:
            cv2.imwrite(f"result{k}.png", B[L-1]*255)

        
