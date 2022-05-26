import geo_modification as gm
import numpy as np
import basics
import queue
import masks
import math
import cv2

#
SLASH, HORI, BSLASH, VERT = 0, 1, 2, 3
#
def tan2orientation(tan):
    (R, C), TAN22_5, TAN67_5, output = tan.shape, math.tan(math.pi/8), math.tan(math.pi*3/8), np.full(tan.shape, VERT)
    for i in range(R):
        for j in range(C):
            if TAN22_5 <= tan[i][j] and tan[i][j] < TAN67_5:
                output[i][j] = SLASH
            elif -TAN22_5 <= tan[i][j] and tan[i][j] < TAN22_5:
                output[i][j] = HORI
            elif -TAN67_5 <= tan[i][j] and tan[i][j] < -TAN22_5:
                output[i][j] = BSLASH
    return output
#
def gradient_magnitude(source, mask):
    (R, C), grad, padd = source.shape, np.zeros(source.shape), basics.padding(source, 3)
    for i, I in zip(range(R), range(3, R+3)):
        for j, J in zip(range(C), range(3, C+3)):
            grad[i][j] = np.sum(padd[i:I, j:J]*mask)
    return grad
#
def noise_reduction(source, window, var):
    (R, C), mask, padd, output = source.shape, masks.Gaussian(window, var), basics.padding(source, window), np.zeros(source.shape)
    for i, I in zip(range(R), range(window, window + R)):
        for j, J in zip(range(C), range(window, window + C)):
            output[i][j] = np.sum(padd[i:I, j:J]*mask)
    return output
#
def non_maximal_suppression(source, orient):
    (R, C), output, padd = source.shape, np.zeros(source.shape), np.pad(source, 1, mode='constant') # zero-padding
    
    for i, I in zip(range(R), range(1, 1 + R)):
        for j, J in zip(range(C), range(1, 1 + C)):
            if orient[i][j] < HORI:
                v_, v__ = padd[I-1][J+1], padd[I+1][J-1]
            elif orient[i][j] < BSLASH:
                v_, v__ = padd[I][J+1], padd[I][J-1]
            elif orient[i][j] < VERT:
                v_, v__ = padd[I+1][J+1], padd[I-1][J-1]
            else:
                v_, v__ = padd[I+1][J], padd[I-1][J]

            if v_ < source[i][j] and v__ < source[i][j]:
                output[i][j] = source[i][j]
    return output
################################################

def Hough_transform(source):
    (R, C) = source.shape
    (x, y) = gm.Image2Cartesian(R, C)
    output = np.zeros((1800, 1800))
    for i in range(R):
        for j in range(C):
            if source[i][j] != 0:
                for theta in range(1, 1800):
                    m = (-1)/math.tan(theta*math.pi/1800)
                    d = int((y[i][j] - m*x[i][j])/math.sqrt(m*m + 1))
                    output[d+800][theta] = 255
    return output
######################################################

#
def Sobel(source, grad_only=False):
    (R, C), Rmask, Cmask = source.shape, masks.RG(2), masks.CG(2)
    gR, gC = gradient_magnitude(source, Rmask), gradient_magnitude(source, Cmask) 
    grad, output = np.sqrt(gR*gR + gC*gC), np.full((R, C), 0)
    thres = np.mean(grad) + np.std(grad)
    
    for i in range(R):
        for j in range(C):
            if grad[i][j] > thres:
                output[i][j] = 255
    if grad_only:
        return grad
    return grad, output
#
def Canny(source):
    # Noise reduction
    (R, C), unnoise = source.shape, noise_reduction(source, 5, 1)  # var ∝ window
    #cv2.imwrite('./noise reduction.png', unnoise)
    
    # Compute gradient magnitude and orientation
    Rmask, Cmask = masks.RG(5), masks.CG(5) # noneffective ?
    gR, gC = gradient_magnitude(unnoise, Rmask) + np.full((R,C), 10e-4), gradient_magnitude(unnoise, Cmask)
    grad, tan = np.sqrt(gR*gR + gC*gC), (gC / gR)
    #cv2.imwrite('./gradient magnitude.png', grad)
    
    # Non-maximal suppression
    orient = tan2orientation(tan)
    nmsup = non_maximal_suppression(grad, orient)
    #cv2.imwrite('./non maximal suppression.png', nmsup)
    
    # hysteretic thresholding
    T = np.copy(nmsup).flatten('C')
    T = np.delete(T, np.argwhere(T == 0))
    hythres, T_L, T_H = np.full((R, C), 255), np.mean(T), np.mean(T) + 0.5*np.std(T)
    for i in range(R):
        for j in range(C):
            if nmsup[i][j] < T_L:
                hythres[i][j] = 0
            elif nmsup[i][j] < T_H:
                hythres[i][j] //= 2
    #cv2.imwrite('./hysteretic thresholding.png', hythres)
    
    # Connected component labeling method
    que = queue.LifoQueue()
    for i in range(R):
        for j in range(C):
            if hythres[i][j] > 127:
                que.put((i,j))
    
    while not que.empty():
        (i, j) = que.get()
        
        if orient[i][j] < HORI:
            i_, j_, i__, j__ = (i - 1), (j - 1), (i + 1), (j + 1)
        elif orient[i][j] < BSLASH:
            i_, j_, i__, j__ = (i - 1), (j), (i + 1), (j)
        elif orient[i][j] < VERT:
            i_, j_, i__, j__ = (i + 1), (j - 1), (i - 1), (j + 1)
        else:
            i_, j_, i__, j__ = (i), (j - 1), (i), (j + 1)
        
        hythres[i][j] = 255
        if 0 <= i_ and i_ < R and 0 <= j_ and j_ < C and hythres[i_][j_] == 127:
            que.put((i_, j_))
        if 0 <= i__ and i__ < R and 0 <= j__ and j__ < C and hythres[i__][j__] == 127:
            que.put((i__, j__))
    return hythres - 127*(hythres == 127)
#
def Laplaican_Gaussian(source, window, var):
    (R, C), mask, padd, output = source.shape, masks.LoG(window, var), basics.padding(source, window), np.zeros(source.shape)
    
    for i, I in zip(range(R), range(window, window + R)):
        for j, J in zip(range(C), range(window, window + C)):
            output[i][j] = np.sum(padd[i:I, j:J]*mask)
    return output
#
def LP_Filtering(source, K):
    (R, C), mask, padd, output = source.shape, masks.LP(K), basics.padding(source, 3), np.copy(source)
    for i, I in zip(range(R), range(3, R+3)):
        for j, J in zip(range(C), range(3, C+3)):
            output[i][j] = np.sum(padd[i:I, j:J]*mask)
    return output
#
def edge_crispening(source):
    unsharp, LoG = LP_Filtering(Sobel(source, True), 1), Laplaican_Gaussian(source, 5, 0.5)
    enhance = ((unsharp - np.mean(unsharp)) / np.std(unsharp))*LoG
    return source + enhance

# ref : https://jason-chen-1992.weebly.com/home/-unsharp-masking

if __name__ == "__main__":
    pass