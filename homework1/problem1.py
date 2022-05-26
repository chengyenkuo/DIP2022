import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

def plot_histogram(source, name):
    plt.hist(source.ravel(), 256, [0, 256])
    plt.savefig('Histogram/' + name)
    plt.clf()

unif, lookup = (np.arange(1., 257., 1) / 256), np.zeros(256)

def global_histogram_equalization(source):
    cdf, bins = np.histogram(source, bins=256, range=[0,256], density=True)
    
    for i in range(1,256):
        cdf[i] += cdf[i-1]
    
    for i in range(256):
        lookup[i] = np.searchsorted(unif, cdf[i])

    R, C, output = source.shape[0], source.shape[1], np.copy(source)
    for i in range(R):
        for j in range(C):
            output[i][j] = lookup[round(output[i][j])]
    return output

def padding(source, window):
    r, c, w, output = source.shape[0], source.shape[1], window//2, np.copy(source)

    L, R = np.flip(output[:,1:(w+1)], axis=1), np.flip(output[:,(c-w):c], axis=1)
    output = np.hstack([L,output,R])
    U, D = np.flip(output[1:(w+1),:], axis=0), np.flip(output[(r-w):r,:], axis=0)
    output = np.vstack([U,output,D])
    return output

def local_histogram_equalization(source, window):
    r, c, a, p, output = source.shape[0], source.shape[1], window*window, padding(source, window), np.copy(source)
    for i in range(r):
        for j in range(c):
            cdf = (p[i:i+window, j:j+window] <= source[i][j]).sum() / a
            output[i][j] = np.searchsorted(unif, cdf)
    return output

# https://www.delftstack.com/zh-tw/api/numpy/python-numpy-numpy.histogram-function/
# https://blog.csdn.net/shizheng_Li/article/details/116015222
# https://www.cnblogs.com/wangjingchn/p/7376470.html

# Problem 1

sample2 = cv2.imread('SampleImage/sample2.png', cv2.IMREAD_GRAYSCALE)
R, C = sample2.shape[0], sample2.shape[1]

# (a)
result3 = sample2 / 2
cv2.imwrite('result3.png', result3)

# (b)
result4 = np.zeros((R,C))
for i in range(R):
    for j in range(C):
        if result3[i][j]*3 > 255:
            result4[i][j] = 255
        else:
            result4[i][j] = result3[i][j]*3
cv2.imwrite('result4.png', result4)

# (c)
plot_histogram(sample2, 'sample2.png')
plot_histogram(result3, 'result3.png')
plot_histogram(result4, 'result4.png')

# (d)
result5 = global_histogram_equalization(result3)
cv2.imwrite('result5.png', result5)
plot_histogram(result5, 'result5.png')

result6 = global_histogram_equalization(result4)
cv2.imwrite('result6.png', result6)
plot_histogram(result6, 'result6.png')


# (e)
result7 = local_histogram_equalization(result3, 101)
cv2.imwrite('result7.png', result7)
plot_histogram(result7, 'result7.png')

result8 = local_histogram_equalization(result4, 101)
cv2.imwrite('result8.png', result8)
plot_histogram(result8, 'result8.png')       

# (f)
window = 100
result9 = np.copy(sample2)
for i in range(0, R, 10):
    for j in range(0, C, 10):
        result9[i:i+window, j:j+window] = global_histogram_equalization(sample2[i:i+window, j:j+window])
cv2.imwrite('result9.png', result9)
plot_histogram(result9, 'result9.png')