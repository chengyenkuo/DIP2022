import numpy as np
import math
import cv2

def low_pass(frequency, threshold=20):
    R, C=frequency.shape
    #cv2.imwrite('frequency.png', 20*np.abs(np.log(frequency)))
    for r in range(R):
        for c in range(C):
            if ((r - R/2)**2 + (c - C/2)**2)**(1/2) > threshold:
                frequency[r][c]=0
    LP=np.copy(frequency)
    for r in range(R):
        for c in range(C):
            if LP[r][c] != 0:
                LP[r][c]=np.log(LP[r][c])
    #cv2.imwrite('LP.png', 20*np.abs(LP))
    return frequency

def unsharp_masking(source, c=3/5):
    # Low-pass (from spacial domain to frequency domain with shifting)
    LP=low_pass(np.fft.fftshift(np.fft.fft2(source)))

    # from frequency domain to spacial domain with shifting back
    LP=np.abs(np.fft.ifft2(np.fft.ifftshift(LP)))

    # unsharpen
    return c/(2*c-1)*source - (1-c)/(2*c-1)*LP


if __name__ == "__main__":
    # (a)
    sample2=cv2.imread('sample/sample2.png', cv2.IMREAD_GRAYSCALE)
    result5=cv2.resize(sample2, (500, 500), cv2.INTER_NEAREST)
    result5=cv2.resize(result5, sample2.shape, cv2.INTER_NEAREST)
    cv2.imwrite('result5.png', result5)

    
    # (b)
    sample3=cv2.imread('sample/sample3.png', cv2.IMREAD_GRAYSCALE)
    result6=unsharp_masking(sample3)
    cv2.imwrite('result6.png', result6)



