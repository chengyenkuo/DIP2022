from statistics import mode
import numpy as np
import cv2

def dithering(source, dither_matrix):
    R, C, N, output=np.size(source, 0), np.size(source, 1), np.size(dither_matrix, 0), np.copy(source)

    T=255*(dither_matrix+0.5)/(N**2)
    for r in range(0, R, N):
        for c in range(0, C, N):
            for i in range(N):
                for j in range(N):
                    if source[i+r][j+c] > T[i][j]:
                        output[i+r][j+c] = 255
                    else:
                        output[i+r][j+c] = 0
    return output
            
def expand_dither_matrix(source, I):
    N=np.size(source, 0)
    output, cnt=np.zeros((2*N, 2*N)), 0
    
    Ridxs4I, Cidxs4I=I.flatten().argsort()//2, I.flatten().argsort()%2
    Ridxs4source, Cidxs4source=source.flatten().argsort()//N, source.flatten().argsort()%N
    
    for i, j in zip(Ridxs4source, Cidxs4source):
        for r, c in zip(Ridxs4I*N + i, Cidxs4I*N +j):
            output[r][c], cnt=cnt, cnt+1
    return output.astype(int)
'''
def Floyd_Steinberg_inclass(source, round=1, unidirectional=False):
    (R, C), output=source.shape, np.pad(source, ((0, 1), (1, 1)), mode='constant')
    threshold, mask=127, np.array([[0, 0, 7], [3, 5, 1]])/16
    
    error_defusion=np.zeros((R+1, C+2))
    for k in range(round+1):
        F_tilde, error_defusion=source+error_defusion[:-1, 1:-1], np.zeros((R+1, C+2))
        output=(F_tilde > threshold)*255
        error=F_tilde-output
    
        for r in range(0, R, 1 if unidirectional else 2):
            for c in range(C): # from left to right
                for i in range(2):
                    for j in range(3):
                        error_defusion[r+i][c+i]+=error[r][c]*mask[i][j]
            if unidirectional:
                continue
            r, mask=(r+1), np.flip(mask, 1)
            for c in range((C-1), (-1), (-1)): # from right to left
                for i in range(2):
                    for j in range(3):
                        error_defusion[r+i][c+i]+=error[r][c]*mask[i][j]
            mask=np.flip(mask, 1)
    return output
'''
def Floyd_Steinberg(source, unidirectional=False):
    (R, C), output=source.shape, np.pad(source, ((0, 1), (1, 1)), mode='constant')
    threshold, mask=127, np.array([[0, 0, 7], [3, 5, 1]])/16

    for r in range(0, R, 1 if unidirectional else 2):
        for c in range(C): # from left to right
            old_pixel, output[r][c+1]=output[r][c+1], (255 if output[r][c+1] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+1])    # note the order
            for i in range(2):
                for j in range(3):
                    output[r+i][c+j]+=error*mask[i][j]
        if unidirectional:
            continue
        r, mask=(r+1), np.flip(mask, 1)
        for c in range((C-1), (-1), (-1)): # from right to left
            old_pixel, output[r][c+1]=output[r][c+1], (255 if output[r][c+1] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+1])    # note the order
            for i in range(2):
                for j in range(3):
                    output[r+i][c+j]+=error*mask[i][j]
        mask=np.flip(mask, 1)
    return output[:-1,1:-1]
'''
def Jarvis_inclass(source, round=1, unidirectional=False):
    (R, C), output=source.shape, np.pad(source, ((0, 1), (1, 1)), mode='constant')
    threshold, mask=127, np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]])/48
    
    error_defusion=np.zeros((R+2, C+4))
    for k in range(round+1):
        F_tilde, error_defusion=source+error_defusion[:-2, 2:-2], np.zeros((R+2, C+4))
        output=(F_tilde > threshold)*255
        error=F_tilde-output
    
        for r in range(0, R, 1 if unidirectional else 2):
            for c in range(C): # from left to right
                for i in range(3):
                    for j in range(5):
                        error_defusion[r+i][c+i]+=error[r][c]*mask[i][j]
            if unidirectional:
                continue
            r, mask=(r+1), np.flip(mask, 1)
            for c in range((C-1), (-1), (-1)): # from right to left
                for i in range(3):
                    for j in range(5):
                        error_defusion[r+i][c+i]+=error[r][c]*mask[i][j]
            mask=np.flip(mask, 1)
    return output
'''
def Jarvis(source, unidirectional=False):
    (R, C), output=source.shape, np.pad(source, ((0, 2), (2, 2)), mode='constant')
    threshold, mask=127, np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]])/48

    for r in range(0, R, 1 if unidirectional else 2):
        for c in range(C): # from left to right
            old_pixel, output[r][c+2]=output[r][c+2], (255 if output[r][c+2] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+2])    # note the order
            for i in range(3):
                for j in range(5):
                    output[r+i][c+j]+=error*mask[i][j]
        if unidirectional:
            continue
        r, mask=(r+1), np.flip(mask, 1)
        for c in range((C-1), (-1), (-1)): # from right to left
            old_pixel, output[r][c+2]=output[r][c+2], (255 if output[r][c+2] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+2])    # note the order
            for i in range(3):
                for j in range(5):
                    output[r+i][c+j]+=error*mask[i][j]
        mask=np.flip(mask, 1)
    return output[:-2,2:-2]

def Sierra(source, unidirectional=False):
    (R, C), output=source.shape, np.pad(source, ((0, 2), (2, 2)), mode='constant')
    threshold, mask=127, np.array([[0, 0, 0, 5, 3], [2, 4, 5, 4, 2], [0, 2, 3, 2, 0]])/32

    for r in range(0, R, 1 if unidirectional else 2):
        for c in range(C): # from left to right
            old_pixel, output[r][c+2]=output[r][c+2], (255 if output[r][c+2] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+2])    # note the order
            for i in range(3):
                for j in range(5):
                    output[r+i][c+j]+=error*mask[i][j]
        if unidirectional:
            continue
        r, mask=(r+1), np.flip(mask, 1)
        for c in range((C-1), (-1), (-1)): # from right to left
            old_pixel, output[r][c+2]=output[r][c+2], (255 if output[r][c+2] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+2])    # note the order
            for i in range(3):
                for j in range(5):
                    output[r+i][c+j]+=error*mask[i][j]
        mask=np.flip(mask, 1)
    return output[:-2,2:-2]

def Sierra_Lite(source, unidirectional=False):
    (R, C), output=source.shape, np.pad(source, ((0, 1), (1, 1)), mode='constant')
    threshold, mask=127, np.array([[0, 0, 2], [1, 1, 0]])/4

    for r in range(0, R, 1 if unidirectional else 2):
        for c in range(C): # from left to right
            old_pixel, output[r][c+1]=output[r][c+1], (255 if output[r][c+1] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+1])    # note the order
            for i in range(2):
                for j in range(3):
                    output[r+i][c+j]+=error*mask[i][j]
        if unidirectional:
            continue
        r, mask=(r+1), np.flip(mask, 1)
        for c in range((C-1), (-1), (-1)): # from right to left
            old_pixel, output[r][c+1]=output[r][c+1], (255 if output[r][c+1] > threshold else 0)
            error=int(old_pixel)-int(output[r][c+1])    # note the order
            for i in range(2):
                for j in range(3):
                    output[r+i][c+j]+=error*mask[i][j]
        mask=np.flip(mask, 1)
    return output[:-1,1:-1]

if __name__ == "__main__":
    sample1=cv2.imread('sample/sample1.png', cv2.IMREAD_GRAYSCALE)
    
    # (a)
    I=np.array([[1, 2], [3, 0]])    # dither_matrix
    result1=dithering(sample1, I)
    cv2.imwrite('result1.png', dithering(sample1, I))
    
    # (b)
    dither_matrix=I
    for i in range(7):
        dither_matrix=expand_dither_matrix(dither_matrix, I)
    result2=dithering(sample1, dither_matrix)
    cv2.imwrite('result2.png', result2)
    
    # (c)
    result3=Floyd_Steinberg(sample1, True)
    #cv2.imwrite('result3_uni.png', result3)
    result3=Floyd_Steinberg(sample1)
    cv2.imwrite('result3.png', result3)

    result4=Jarvis(sample1, True)
    #cv2.imwrite('result4_uni.png', result4)
    result4=Jarvis(sample1)
    cv2.imwrite('result4.png', result4)
    '''
    result4=Sierra(sample1, True)
    cv2.imwrite('Sierra_uni.png', result4)
    result4=Sierra(sample1)
    cv2.imwrite('Sierra.png', result4)

    result4=Sierra_Lite(sample1, True)
    cv2.imwrite('Sierra_Lite_uni.png', result4)
    result4=Sierra_Lite(sample1)
    cv2.imwrite('Sierra_Lite.png', result4)
    '''
    # reference: https://blog.csdn.net/WhoisPo/article/details/104689737