import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

peak = 255*255
def PSNR(original, noisy):
    MSE = np.sum(np.square(original - noisy)) / np.prod(original.shape)
    return 10*math.log10(peak/MSE)

def plot_histogram(source, name):
    plt.hist(source.ravel(), 256, [0, 256])
    plt.savefig('Histogram/' + name)
    plt.clf()

def padding(source, window):
    r, c, w, output = source.shape[0], source.shape[1], window//2, np.copy(source)

    L, R = np.flip(output[:,1:(w+1)], axis=1), np.flip(output[:,(c-w):c], axis=1)
    output = np.hstack([L,output,R])
    U, D = np.flip(output[1:(w+1),:], axis=0), np.flip(output[(r-w):r,:], axis=0)
    output = np.vstack([U,output,D])
    return output

def low_pass(source, b):
    r, c, p, output = source.shape[0], source.shape[1], padding(source, 3), np.copy(source)
    H = np.array([[1, b, 1], [b, b*b, b], [1, b, 1]]) / ((b+2)*(b+2))
    print(H)
    for i in range(r):
        for j in range(c):
            output[i][j] = np.sum(np.multiply(p[i:i+3, j:j+3], H))
    return output


def outlier_detection(source, window):
    sigma, output, filter = np.sqrt(np.var(source)), np.copy(source), np.ones((window, window))
    # sigma = np.sqrt(np.sum(np.square(source - np.mean(source)))/np.size(source))
    
    R, C, N, P = source.shape[0], source.shape[1], (window*window - 1), padding(source, window)
    
    filter[window//2][window//2] = 0
    for i in range(R):
        for j in range(C):
            ave = np.sum(np.multiply(P[i:(i+window), j:(j+window)], filter)) / N
            if abs(output[i][j] - ave) > sigma:
                output[i][j] = ave
    return output

def median_filtering(source, window):
    R, C, P, output = source.shape[0], source.shape[1], padding(source, window), np.copy(source)

    for i in range(R):
        for j in range(C):
            output[i][j] = np.median(P[i:(i+window), j:(j+window)])
    return output

def MAXMIN(source, window):
    r, c, w = source.shape[0], source.shape[1], window // 2

    P, output = padding(source, window), np.copy(source)
    for i in range(r):
        for j in range(c):
            max = 0
            for k in range(i,i+w+1):
                for l in range(j,j+w+1):
                    local_min = np.min(P[k:k+w, l:l+w])
                    if local_min > max:
                        max = local_min
            output[i][j] = max
    return output

def MINMAX(source, window):
    r, c, w = source.shape[0], source.shape[1], window // 2

    P, output = padding(source, window), np.copy(source)
    for i in range(r):
        for j in range(c):
            min = 256
            for k in range(i,i+w+1):
                for l in range(j,j+w+1):
                    local_max = np.max(P[k:k+w, l:l+w])
                    if local_max < min:
                        min = local_max
            output[i][j] = min
    return output

def PMED(source, window):
    maxmin, minmax = MAXMIN(source, window), MINMAX(source, window)
    return (maxmin + minmax) / 2

# Problem 2
sample3 = cv2.imread('SampleImage/sample3.png', cv2.IMREAD_GRAYSCALE)
plot_histogram(sample3, 'sample3.png')
sample4 = cv2.imread('SampleImage/sample4.png', cv2.IMREAD_GRAYSCALE)
plot_histogram(sample4, 'sample4.png')
sample5 = cv2.imread('SampleImage/sample5.png', cv2.IMREAD_GRAYSCALE)
plot_histogram(sample5, 'sample5.png')

#print(PSNR(sample3,sample4))
#print(PSNR(sample3,sample5))

result10 = MAXMIN(MINMAX(sample4, 3), 3)
plot_histogram(result10, 'result10.png')
cv2.imwrite('result10.png', result10)
#print(PSNR(sample3, result10))

result11 = median_filtering(sample5, 3)
plot_histogram(result11, 'result11.png')
cv2.imwrite('result11.png', result11)
#print(PSNR(sample3, result11))