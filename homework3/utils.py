import os
import math
import numpy as np
import cv2

def erosion(img, H, window):
    img = np.pad(img, ((window, window), (window, window)), mode='constant', constant_values=0)
    tmp = np.zeros((img.shape[0], img.shape[1]))
    print(tmp.shape)
    cnt = 0
    for i in range(window):
        for j in range(window):
            if H[i][j] and cnt == 0:
                for ii in range(window, img.shape[0] - window):
                    for jj in range(window, img.shape[1] - window):
                        tmp[ii + i][jj + j] = img[ii][jj]
                cnt += 1
            elif H[i][j]:
                for ii in range(window, img.shape[0] - window):
                    for jj in range(window, img.shape[1] - window):
                        if tmp[ii + i][jj + j] == 255:
                            tmp[ii + i][jj + j] = 255 if img[ii][jj] == 255 else 0
    
    return tmp[window + 1 : img.shape[0] - window + 1 , window + 1 : img.shape[0] - window + 1]

def complement(img):
    ans = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ans[i][j] = 0 if img[i][j] > 128 else 255
            
    return ans
    
def Hole_Filling(img, img_c):
    import queue
    q = queue.Queue()
    start = [(247, 103), (239, 131), (283, 123), (432, 207), (406, 270), (500, 237), (457, 292), (473, 297), (488, 304)]
    for point in start:
        q.put((point[0], point[1]))
    cnt = 0
    while cnt < 4391:
        p = q.get()
        a, b = p[0], p[1]
        if img_c[a-1][b] == 255 and img[a-1][b] == 0:
            q.put((a-1, b))
            img[a-1][b] = 255
        if img_c[a][b-1] == 255 and img[a][b-1] == 0:
            q.put((a, b-1))
            img[a][b-1] = 255
        if img_c[a][b+1] == 255 and img[a][b+1] == 0:
            q.put((a, b+1))
            img[a][b+1] = 255
        if img_c[a+1][b] == 255 and img[a+1][b] == 0:
            q.put((a+1, b))
            img[a+1][b] = 255
        cnt += 1
        #print(cnt)
    return img
        
def delete_countor_1(img):
    tool1 = [-1, 0, 1]
    img = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    res = img.copy()
    change = 0
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == 255:
                cnt = 0
                flag = 0
                # step 1 kick B(P)
                for ii in tool1:
                    for jj in tool1:
                        if img[i + ii, j + jj] == 255:
                            cnt += 1
                if cnt >= 3 and cnt <= 7:
                    flag += 1
                # step 2 kick A(P)
                tool2 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
                cnt = 0
                for index in range(8):
                    a1, b1 = tool2[index][0], tool2[index][1]
                    a2, b2 = tool2[index+1][0], tool2[index+1][1]
                    if img[i+a1, j+b1] == 0 and img[i+a2, j+b2] == 255:
                        cnt += 1
                if cnt == 1:
                    flag += 1
                # step 3 P2*P4*P6 -> (-1, 0), (0, 1), (1, 0)
                if img[i-1][j] == 0 or img[i][j+1] == 0 or img[i+1][j] == 0:
                    flag += 1
                # step 4 P4*P6*P8 -> (0, 1), (1, 0), (0, -1)
                if img[i][j+1] == 0 or img[i+1][j] == 0 or img[i][j-1] == 0:
                    flag += 1
                if flag == 4:
                    res[i][j] = 0
                    change += 1
    print('iter 1 change = {}'.format(change))
    return res[1 : res.shape[0] - 1, 1 : res.shape[1] - 1], change

def delete_countor_2(img):
    tool1 = [-1, 0, 1]
    img = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    res = img.copy()
    change = 0
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == 255:
                flag = 0
                # step 1 kick B(P)
                cnt = 0
                for ii in tool1:
                    for jj in tool1:
                        cnt += img[i + ii, j + jj]
                if cnt >= 255 * 3 and cnt <= 255 * 7:
                    flag += 1
                # step 2 kick A(P)
                tool2 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
                cnt = 0
                for index in range(8):
                    a1, b1 = tool2[index][0], tool2[index][1]
                    a2, b2 = tool2[index+1][0], tool2[index+1][1]
                    if img[i+a1, j+b1] == 0 and img[i+a2, j+b2] == 255:
                        cnt += 1
                if cnt == 1:
                    flag += 1
                # step 3 P2*P4*P8 -> (-1, 0), (0, 1), (0, -1)
                if img[i-1][j] == 0 or img[i][j+1] == 0 or img[i][j-1] == 0:
                    flag += 1
                # step 4 P2*P6*P8 -> (-1, 0), (1, 0), (0, -1)
                if img[i-1][j] == 0 or img[i+1][j] == 0 or img[i][j-1] == 0:
                    flag += 1
                if flag == 4:
                    res[i][j] = 0
                    change += 1
    print('iter 2 change = {}'.format(change))
    return res[1 : res.shape[0] - 1, 1 : res.shape[1] - 1], change
    
def labeling(img, p, color):
    import queue
    q = queue.Queue()
    q.queue.clear()
    img[p[0]][p[1]] = color
    q.put(p)
    tool = [-1, 0, 1]
    white = np.array([255, 255, 255])
    cnt = 0
    while not q.empty():
        #print(q.qsize())
        point = q.get()
        a1, a2 = point[0], point[1]
        for i in tool:
            for j in tool:
                if i == j == 0:
                    continue
                if np.array_equal(img[a1+i][a2+j], white):
                    img[a1+i][a2+j] = color
                    q.put((a1+i, a2+j))
                    cnt += 1
        #print(cnt)
    return img

def filtering(img, filter, window):
    res = np.zeros((img.shape[0], img.shape[1]))
    shift = window // 2
    img1 = np.pad(img, ((shift, shift), (shift, shift)), 'edge')
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            tmp = 0.
            for st in range(window):
                for end in range(window):
                    tmp += img1[i + st][j + end] * filter[st][end]
            res[i][j] = tmp
    return res
    
def sq_filtering(img, filter, window):
    res = np.zeros((img.shape[0], img.shape[1]))
    shift = window // 2
    img1 = np.pad(img, ((shift, shift), (shift, shift)), 'edge')
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            tmp = 0.
            for st in range(window):
                for end in range(window):
                    tmp += (img1[i + st][j + end] ** 2) * filter[st][end]
            res[i][j] = tmp
    return res

def make_filter():
    a1 = np.array([[1,2,1],[2,4,2],[1,2,1]]) * 1/36
    a2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) * 1/12
    a3 = np.array([[-1,2,-1],[-2,4,-2],[-1,2,-1]]) * 1/12
    a4 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) * 1/12
    a5 = np.array([[1,0,-1],[0,0,0],[-1,0,1]]) * 1/4
    a6 = np.array([[-1,2,-1],[0,0,0],[1,-2,1]]) * 1/4
    a7 = np.array([[-1,-2,-1],[2,4,2],[-1,-2,-1]]) * 1/12
    a8 = np.array([[-1,0,1],[2,0,-2],[-1,0,1]]) * 1/4
    a9 = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]]) * 1/4
    return [a1, a2, a3, a4, a5, a6, a7, a8, a9]
    
def making_data(lists):
    data = []
    for i in range(600):
        for j in range(900):
            tmp = []
            for list in lists:
                #print(i, j, list[i][j])
                tmp.append(list[i][j])
            data.append(tmp)
    return np.asarray(data)

def label_pairing(datas, colors):
    final = np.zeros((600, 900))
    filling = [0, 85, 170, 255]
    for color, fill in zip(colors, filling):
        for index in color:
            a, b = index // 900, index % 900
            final[a, b] = fill
    return final
def k_means(datas):
    print(datas.shape)
    pre_k1, pre_k2, pre_k3, pre_k4 = np.random.rand(9)*10, np.random.rand(9)*10, np.random.rand(9)*10, np.random.rand(9)*10
    k1, k2, k3, k4 = datas[18041], datas[178053], datas[353893], datas[495404]
    cnt = 0
    final_color = []
    while not (np.array_equal(k1, pre_k1) and np.array_equal(k2, pre_k2) and np.array_equal(k3, pre_k3) and np.array_equal(k4, pre_k4)):
        color = [[], [], [], []]
        for i, data in enumerate(datas):
            dis1 = np.sqrt(np.sum(np.square(data-k1)))
            dis2 = np.sqrt(np.sum(np.square(data-k2)))
            dis3 = np.sqrt(np.sum(np.square(data-k3)))
            dis4 = np.sqrt(np.sum(np.square(data-k4)))
            #print(dis1, dis2, dis3, dis4)
            index = np.argmin([dis1, dis2, dis3, dis4])
            color[index].append(i)
        tmp1 = [datas[i] for i in color[0]]
        pre_k1 = k1
        k1 = np.zeros(9) if len(tmp1) == 0 else 1 / len(tmp1) * np.sum(np.array(tmp1), axis=0)
        tmp2 = [datas[i] for i in color[1]]
        pre_k2 = k2
        k2 = np.zeros(9) if len(tmp2) == 0 else 1 / len(tmp2) * np.sum(np.array(tmp2), axis=0)
        tmp3 = [datas[i] for i in color[2]]
        pre_k3 = k3
        k3 = np.zeros(9) if len(tmp3) == 0 else 1 / len(tmp3) * np.sum(np.array(tmp3), axis=0)
        tmp4 = [datas[i] for i in color[3]]
        pre_k4 = k4
        k4 = np.zeros(9) if len(tmp4) == 0 else 1 / len(tmp4) * np.sum(np.array(tmp4), axis=0)
        print('iteration {} is finished !!!'.format(cnt + 1))
        cnt += 1
        final_color = color
        print(len(final_color[0]), len(final_color[1]), len(final_color[2]), len(final_color[3]))
    print('k means clustering done!!!')
    
    return label_pairing(datas, final_color)
    
def aitchison(u, v):
    log_u_v = np.log(u / v)
    dist = np.linalg.norm(log_u_v - np.mean(log_u_v))
    return dist
    
def k_means_aitch(datas):
    print(datas.shape)
    pre_k1, pre_k2, pre_k3, pre_k4 = np.random.rand(9)*10, np.random.rand(9)*10, np.random.rand(9)*10, np.random.rand(9)*10
    k1, k2, k3, k4 = datas[18041], datas[178053], datas[353893], datas[495404]
    cnt = 0
    final_color = []
    while not (np.array_equal(k1, pre_k1) and np.array_equal(k2, pre_k2) and np.array_equal(k3, pre_k3) and np.array_equal(k4, pre_k4)):
        color = [[], [], [], []]
        for i, data in enumerate(datas):
            dis1 = aitchison(data, k1)
            dis2 = aitchison(data, k2)
            dis3 = aitchison(data, k3)
            dis4 = aitchison(data, k4)
            #print(dis1, dis2, dis3, dis4)
            index = np.argmin([dis1, dis2, dis3, dis4])
            color[index].append(i)
        tmp1 = [datas[i] for i in color[0]]
        pre_k1 = k1
        k1 = np.zeros(9) if len(tmp1) == 0 else 1 / len(tmp1) * np.sum(np.array(tmp1), axis=0)
        tmp2 = [datas[i] for i in color[1]]
        pre_k2 = k2
        k2 = np.zeros(9) if len(tmp2) == 0 else 1 / len(tmp2) * np.sum(np.array(tmp2), axis=0)
        tmp3 = [datas[i] for i in color[2]]
        pre_k3 = k3
        k3 = np.zeros(9) if len(tmp3) == 0 else 1 / len(tmp3) * np.sum(np.array(tmp3), axis=0)
        tmp4 = [datas[i] for i in color[3]]
        pre_k4 = k4
        k4 = np.zeros(9) if len(tmp4) == 0 else 1 / len(tmp4) * np.sum(np.array(tmp4), axis=0)
        print('iteration {} is finished !!!'.format(cnt + 1))
        cnt += 1
        final_color = color
        print(len(final_color[0]), len(final_color[1]), len(final_color[2]), len(final_color[3]))
    print('k means clustering done!!!')
    
    return label_pairing(datas, final_color)
    
def cosine(u, v):
    ans = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return ans
def k_means_cosine(datas):
    print(datas.shape)
    pre_k1, pre_k2, pre_k3, pre_k4 = np.random.rand(9)*10, np.random.rand(9)*10, np.random.rand(9)*10, np.random.rand(9)*10
    k1, k2, k3, k4 = datas[18041], datas[178053], datas[353893], datas[495404]
    cnt = 0
    final_color = []
    while not (np.array_equal(k1, pre_k1) and np.array_equal(k2, pre_k2) and np.array_equal(k3, pre_k3) and np.array_equal(k4, pre_k4)):
        color = [[], [], [], []]
        for i, data in enumerate(datas):
            dis1 = cosine(data, k1)
            dis2 = cosine(data, k2)
            dis3 = cosine(data, k3)
            dis4 = cosine(data, k4)
            #print(dis1, dis2, dis3, dis4)
            index = np.argmax([dis1, dis2, dis3, dis4])
            color[index].append(i)
        tmp1 = [datas[i] for i in color[0]]
        pre_k1 = k1
        k1 = np.zeros(9) if len(tmp1) == 0 else 1 / len(tmp1) * np.sum(np.array(tmp1), axis=0)
        tmp2 = [datas[i] for i in color[1]]
        pre_k2 = k2
        k2 = np.zeros(9) if len(tmp2) == 0 else 1 / len(tmp2) * np.sum(np.array(tmp2), axis=0)
        tmp3 = [datas[i] for i in color[2]]
        pre_k3 = k3
        k3 = np.zeros(9) if len(tmp3) == 0 else 1 / len(tmp3) * np.sum(np.array(tmp3), axis=0)
        tmp4 = [datas[i] for i in color[3]]
        pre_k4 = k4
        k4 = np.zeros(9) if len(tmp4) == 0 else 1 / len(tmp4) * np.sum(np.array(tmp4), axis=0)
        print('iteration {} is finished !!!'.format(cnt + 1))
        cnt += 1
        final_color = color
        print(len(final_color[0]), len(final_color[1]), len(final_color[2]), len(final_color[3]))
    print('k means clustering done!!!')
    
    return label_pairing(datas, final_color)
    
def k_means_binary(datas):
    print(datas.shape)
    k = 1
    final_color = []
    mask = []
    for i in range(0, 540000):
        mask.append(i)
    while k != 4:
        color_list = []
        pre_k1, pre_k2 = np.zeros(9), np.zeros(9)
        # 18041 495404, 18041 353893, 18041, 178053
        if k == 1:
            k1, k2 = datas[18041], datas[495404]
        else:
            k1, k2 = np.random.rand(9)*10, np.random.rand(9)*10
        cnt = 0
        print(len(mask))
        while not (np.array_equal(k1, pre_k1) and np.array_equal(k2, pre_k2)):
            color = [[], []]
            for i in mask:
                dis1 = np.sqrt(np.sum(np.square(datas[i]-k1)))
                dis2 = np.sqrt(np.sum(np.square(datas[i]-k2)))
                #print(dis1, dis2, dis3, dis4)
                index = np.argmin([dis1, dis2])
                color[index].append(i)
            tmp1 = [datas[i] for i in color[0]]
            pre_k1 = k1
            k1 = np.zeros(9) if len(tmp1) == 0 else 1 / len(tmp1) * np.sum(np.array(tmp1), axis=0)
            tmp2 = [datas[i] for i in color[1]]
            pre_k2 = k2
            k2 = np.zeros(9) if len(tmp2) == 0 else 1 / len(tmp2) * np.sum(np.array(tmp2), axis=0)
            print('iteration {} is finished !!!'.format(cnt + 1))
            cnt += 1
            color_list = color
            print(len(color[0]), len(color[1]))
        print('k means clustering done!!!')
        v1, v2 = 0, 0
        for i in color_list[0]:
            v1 += np.sqrt(np.sum(np.square(datas[i]-k1)))
        v1 /= len(color_list[0])
        for i in color_list[1]:
            v2 += np.sqrt(np.sum(np.square(datas[i]-k2)))
        v2 /= len(color_list[1])
        print(v1, v2)
        print(len(color_list[0]), len(color_list[1]))
        if v1 < v2:
            final_color.append(color_list[0])
            mask = []
            for i in color_list[1]:
                mask.append(i)
        else:
            final_color.append(color_list[1])
            mask = []
            for i in color_list[0]:
                mask.append(i)
        k += 1
    
    return label_pairing(datas, final_color)
        
                
