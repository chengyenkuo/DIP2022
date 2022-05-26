import texture_analysis as ta
import numpy as np
import basics
import cv2

if __name__ == "__main__":
    sample2 = cv2.imread('./SampleImage/sample2.png', cv2.IMREAD_GRAYSCALE)
    
    # (a)
    sample2=cv2.imread('./SampleImage/sample2.png', cv2.IMREAD_GRAYSCALE)
    np.save('energy_maps', ta.Laws(sample2))
    E, K=np.swapaxes(np.load('energy_maps.npy'), 0, 1), 4
    result6=ta.k_means(K, np.swapaxes(E, 1, 2))
    cv2.imwrite('result6.png', result6)
    
    