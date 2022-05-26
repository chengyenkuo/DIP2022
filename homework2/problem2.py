import geo_modification as gm
import edge_detection as ed
import numpy as np
import basics
import masks
import math
import cv2

if __name__ == "__main__":
    sample3 = cv2.imread('./SampleImage/sample3.png', cv2.IMREAD_GRAYSCALE)
    
    # (a)
    mask, padd, output = masks.LoG(3, 0.5), basics.padding(sample3, 3), np.copy(sample3)
    
    T_L, T_H = 100, 160
    for i, I in zip(range(600), range(3, 3 + 600)):
        for j, J in zip(range(600), range(3, 3 + 600)):
            if output[i][j] > T_L:
                mask = (padd[i:I, j:J] < T_H)
                if np.sum(mask) > 0:
                    output[i][j] = np.sum((padd[i:I, j:J]*mask) / np.sum(mask))
                else:
                    output[i][j] *= 0
    output = output*(225/np.max(output))
    cv2.imwrite('./sample3_improved.png', output)
    
    # (b)
    result7 = np.zeros((600, 600))
    ''''''
    Rin, Cin, Rout, Cout = 600, 600, 350, 350
    x, y = gm.Image2Cartesian(Rout, Cout)
    x, y = gm.translation(x, y, -105, 0)    # step 2
    x, y = gm.scaling(x, y, 7/12, 7/12)     # step 1
    p, q = gm.Cartesian2Image(x, y, Rin, Cin)
    result7[:300, 220:380] = gm.mapping(sample3, p, q)[20:320, 0:160]
    ''''''
    ''''''
    Rin, Cin, Rout, Cout = 600, 600, 600, 600
    x, y = gm.Image2Cartesian(Rout, Cout)
    x, y = gm.rotation(x, y, -math.pi/2)
    p, q = gm.Cartesian2Image(x, y, Rin, Cin)
    result7 += gm.mapping(result7, p, q)
    ''''''
    ''''''
    Rin, Cin, Rout, Cout = 600, 600, 600, 600
    x, y = gm.Image2Cartesian(Rout, Cout)
    x, y = gm.rotation(x, y, -math.pi)
    p, q = gm.Cartesian2Image(x, y, Rin, Cin)
    result7 += gm.mapping(result7, p, q)
    ''''''
    cv2.imwrite('./result7.png', result7)
    
    # (c)
    sample5 = cv2.imread('./SampleImage/sample5.png', cv2.IMREAD_GRAYSCALE)
    result8 = gm.warping(sample5, 25)
    cv2.imwrite("./result8.png", result8)