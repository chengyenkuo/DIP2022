import os
import math
import numpy as np
import cv2
import utils as U
import random
data_dir = './SampleImage'
out_dir = './'
def problem2_a():
    img = cv2.imread(os.path.join(data_dir, 'sample2.png'), cv2.IMREAD_GRAYSCALE)
    filter_list = U.make_filter()
    img_list = []
    for i, filter in enumerate(filter_list):
        print("iteration : {} / 9".format(i+1))
        tmp = U.filtering(img, filter, 3).copy()
        img_list.append(tmp)
    return img_list
    
def problem2_c():
    img_list = problem2_a()
    #for i in img_list:
    #    print(i)
    # shape n * 9
    filter = np.ones((15, 15))
    energy_list = []
    for k, img in enumerate(img_list):
        print('now making the energy {} / 9'.format(k+1))
        tmp = 1 / (15 * 15) * U.sq_filtering(img, filter, 15).copy()
        energy_list.append(tmp)
    data = U.making_data(energy_list)
    #for i in range(540000):
    #    print(data[i])
    final = U.k_means(data)
    cv2.imwrite(os.path.join(out_dir, 'result7.png'), final)

if __name__ == '__main__':
    problem2_c()
    
