import morpho_processing as mp
import numpy as np
import basics
import math
import cv2


if __name__ == "__main__":
    sample1=cv2.imread('./SampleImage/sample1.png', cv2.IMREAD_GRAYSCALE)
    
    # (a)
    erosion1=basics.b2g(mp.erosion(basics.g2b(sample1)))
    #cv2.imwrite('./result1.png', (sample1 - erosion1))
    
    # (b)
    # preprocess
    
    sample1[41][200:300] = 255
    sample1[41][200:380] = 255
    sample1[43][225] = 255  # D
    sample1[43][245] = 255  # I
    sample1[43][265] = 255  # P
    sample1[43][315] = 255  # 2
    sample1[43][335] = 255  # 0
    sample1[43][355] = 255  # 2
    sample1[43][375] = 255  # 2
    sample1[111][272:327] = 255
    sample1[99][301] = 255     # HW3
    sample1[250][105] = 255    # L
    sample1[240][135] = 255    # R
    sample1[280][135] = 255    # smile
    sample1[420][200] = 255    # +
    sample1[405][275] = 255    # -
    sample1[500][225] = 255    # x
    sample1[475][300] = 255    # /
    sample1[495][305] = 255    # /
    cv2.imwrite('salts.png', sample1)
    
    A, Ac, G = basics.g2b(sample1), basics.NOT(basics.g2b(sample1)), np.zeros(sample1.shape)
    
    cnt = 0
    G[43][225] = 1
    G[43][245] = 1
    G[43][265] = 1
    G[43][315] = 1
    G[43][335] = 1
    G[43][355] = 1
    G[43][375] = 1
    G[99][301] = 1
    G[250][105] = 1
    G[240][135] = 1
    G[280][135] = 1
    G[420][200] = 1
    G[405][275] = 1
    G[500][225] = 1
    G[475][300] = 1
    G[495][305] = 1
    while cnt < 100:
        print(cnt)
        g = basics.AND(mp.dilation(G), Ac)
        if np.array_equal(G, g):
            break
        cnt += 1
        G = g
    cv2.imwrite('./result2.png', basics.b2g(basics.OR(A, G)))
    
    # (c)
    preprocess=basics.g2b(sample1)
    skeleton=mp.skeletonizing(preprocess)
    cv2.imwrite('./result3.png', basics.b2g(skeleton))
    preprocess=basics.NOT(basics.g2b(sample1))
    skeleton=mp.skeletonizing(preprocess)
    cv2.imwrite('./result4.png', basics.b2g(skeleton))
    
    # (d)
    (R, C), ccl=sample1.shape, mp.CCL(basics.g2b(sample1))
    numOfColors=len(np.unique(ccl.flatten()))
    print(numOfColors)
    
    BGR=[
        [0, 0, 0], [0, 0, 255], [0, 165, 255], [255, 0, 255],
        [0, 69, 255], [31, 23, 176], [80, 127, 255], [20, 192, 155],
        
        [0, 255, 255], [0, 255, 0], [64, 145, 61], [34, 139, 34],
        [35, 142, 107], [127, 255, 0], [15, 94, 56], [208, 224, 64],
        
        [255, 127, 0], [255, 0, 0], [255, 0, 139], [171, 89, 61],
        [140, 199, 0], [112, 25, 25], [255, 144, 30], [255, 255, 255],
        
        [0, 255, 255], [0, 255, 0], [64, 145, 61], [34, 139, 34],
        [35, 142, 107], [127, 255, 0], [15, 94, 56], [208, 224, 64],[112, 25, 25]
    ]
    result5=np.zeros((R, C, 3))
    for i in range(R):
        for j in range(C):
            for k in range(3):
                result5[i][j][k] = BGR[ccl[i][j]][k]
    cv2.imwrite('./result5.png', result5)
    
    
    