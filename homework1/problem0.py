import matplotlib.pyplot as plt
import numpy as np
import cv2

# Problem 0
sample1 = cv2.imread('SampleImage/sample1.png')

C, result1 = sample1.shape[1], np.zeros(sample1.shape)
for j in range(C):
    result1[:,j,:] = sample1[:,(C-1-j),:]
cv2.imwrite('result1.png', result1)

result2 = result1[:,:,2]*0.299 + result1[:,:,1]*0.587 + result1[:,:,0]*0.114
cv2.imwrite('result2.png', result2)