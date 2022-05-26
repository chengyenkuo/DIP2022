import edge_detection as ed
import numpy as np
import masks
import math
import cv2

if __name__ == "__main__":
    sample1 = cv2.imread('./SampleImage/sample1.png', cv2.IMREAD_GRAYSCALE)
    # (a)
    result1, result2 = ed.Sobel(sample1)
    cv2.imwrite('./result1.png', result1)
    cv2.imwrite('./result2.png', result2)
    
    # (b)
    result3 = ed.Canny(sample1)
    cv2.imwrite('./result3.png', result3)
    
    # (c)
    result4 = ed.Laplaican_Gaussian(sample1, 5, 0.5)    # var ? window
    cv2.imwrite('./result4.png', result4)
    
    # (d)
    sample2 = cv2.imread('./SampleImage/sample2.png', cv2.IMREAD_GRAYSCALE)
    result5 = ed.edge_crispening(sample2)
    cv2.imwrite('./result5.png', result5)
    
    # (e)
    result3 = cv2.imread('./result3.png', cv2.IMREAD_GRAYSCALE)
    result6 = ed.Hough_transform(result3)
    cv2.imwrite('./result6.png', result6)
    