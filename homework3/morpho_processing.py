import numpy as np
import basics
import cv2

def erosion(source):
    pad, out=np.pad(source, ((1,1), (1,1)), constant_values=1), np.zeros(source.shape)
    (R, C), H=source.shape, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    
    for i in range(R):
        for j in range(C):
            if np.array_equal(pad[i:i+3, j:j+3], H):
                out[i][j]=1
    return out

def dilation(source):
    pad, out=np.pad(source, ((1,1), (1,1)), constant_values=0), np.copy(source)
    (R, C), H=source.shape, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    
    for i in range(R):
        for j in range(C):
            if np.any(pad[i:i+3, j:j+3] == H):
                out[i][j]=1
    return out.astype(int)


def zero_ones(source):
    count=0
    if not source[0][1] and source[0][2]:   count+=1
    if not source[0][2] and source[1][2]:   count+=1
    if not source[1][2] and source[2][2]:   count+=1
    if not source[2][2] and source[2][1]:   count+=1
    if not source[2][1] and source[2][0]:   count+=1
    if not source[2][0] and source[1][0]:   count+=1
    if not source[1][0] and source[0][0]:   count+=1
    if not source[0][0] and source[0][1]:   count+=1
    return count

def skeletonizing(source):
    (R, C), IT=source.shape, np.copy(source)
    dr, dc=[-1, -1, 0, 0, 1, 2, 2, 2, 1, 0], [-1, -1, 1, 2, 2, 2, 1, 0, 0, 0]
    
    while True:
        print('*')
        # 1st
        pad, M, count=np.pad(IT, ((1,1), (1,1)), constant_values=0), np.zeros(IT.shape).astype(np.uint8), 0
        for r in range(R):
            for c in range(C):
                if IT[r][c]:
                    if (np.sum(pad[r:r+3, c:c+3]) - IT[r][c] < 2) or (np.sum(pad[r:r+3, c:c+3]) - IT[r][c] > 6):
                        continue
                    if zero_ones(pad[r:r+3, c:c+3]) != 1:
                        continue
                    if pad[r+dr[2]][c+dc[2]] * pad[r+dr[4]][c+dc[4]] * pad[r+dr[6]][c+dc[6]]:
                        continue
                    if pad[r+dr[8]][c+dc[8]] * pad[r+dr[4]][c+dc[4]] * pad[r+dr[6]][c+dc[6]]:
                        continue
                    M[r][c]=1
        count+=np.sum(M)
        IT-=M
        if not count:
            return IT.astype(np.uint8)
        print('*')
        # 2nd
        pad, M, count=np.pad(IT, ((1,1), (1,1)), constant_values=0), np.zeros(IT.shape).astype(np.uint8), 0
        for r in range(R):
            for c in range(C):
                if IT[r][c]:
                    if (np.sum(pad[r:r+3, c:c+3]) - IT[r][c] < 2) or (np.sum(pad[r:r+3, c:c+3]) - IT[r][c] > 6):
                        continue
                    if zero_ones(pad[r:r+3, c:c+3]) != 1:
                        continue
                    if pad[r+dr[2]][c+dc[2]] * pad[r+dr[4]][c+dc[4]] * pad[r+dr[8]][c+dc[8]]:
                        continue
                    if pad[r+dr[2]][c+dc[2]] * pad[r+dr[6]][c+dc[6]] * pad[r+dr[8]][c+dc[8]]:
                        continue
                    M[r][c]=1
        count+=np.sum(M)
        IT-=M
        if not count:
            return IT.astype(np.uint8)

def CCL(source):   # connected component labeling
    (R, C), L=source.shape, 0
    pad=np.pad(source, ((1,1), (1,1)), constant_values=0).astype(int)
    
    equiv=np.zeros(300).astype(int)
    mask=np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]])
    for i, I in zip(range(R), range(1, R+1)):
        for j, J in zip(range(C), range(1, C+1)):
            if pad[I][J]:
                f=(pad[i:i+3, j:j+3]*mask).flatten()
                labels=f[f != 0]
                
                if len(labels) < 1: # no neighbor
                    L+=1
                    equiv[L], pad[I][J]=L, L
                    continue
                
                pad[I][J]=np.min(labels)
                for l in labels:
                    #while equiv[val] != val:
                        #val=equiv[val]
                    if pad[I][J] < equiv[l]:
                        equiv[l]=pad[I][J]
    
    for l in range(equiv.shape[0]):
        val=equiv[l]
        if equiv[val] != val:
            val=equiv[val]
        equiv[l]=val
    
    out=pad[1:R+1, 1:C+1]
    for r in range(R):
        for c in range(C):
            if equiv[out[r][c]] != out[r][c]:
                out[r][c] = equiv[out[r][c]]

    labels, label2val=np.unique(out.flatten()), {}
    for l in labels:
        label2val[l] = np.where(labels == l)[0]
    for r in range(R):
        for c in range(C):
            out[r][c]=label2val[out[r][c]]
    return out.astype(np.uint8)

################################################
  
if __name__ == "__main__":
    sample1=cv2.imread('./SampleImage/sample1.png', cv2.IMREAD_GRAYSCALE)
    
    (R, C), ccl=sample1.shape, CCL(basics.g2b(sample1))
    numOfColors=len(np.unique(ccl.flatten()))
    
    BGR=[
        [0, 0, 0], [0, 0, 255], [0, 165, 255], [255, 0, 255],
        [0, 69, 255], [31, 23, 176], [80, 127, 255], [20, 192, 155],
        
        [0, 255, 255], [0, 255, 0], [64, 145, 61], [34, 139, 34],
        [35, 142, 107], [127, 255, 0], [15, 94, 56], [208, 224, 64],
        
        [255, 127, 0], [255, 0, 0], [255, 0, 139], [171, 89, 61],
        [140, 199, 0], [112, 25, 25], [255, 144, 30], [255, 255, 255]
    ]
    result5=np.zeros((R, C, 3))
    for i in range(R):
        for j in range(C):
            for k in range(3):
                result5[i][j][k] = BGR[ccl[i][j]][k]
    cv2.imwrite('./result5.png', result5)
    
    exit(0)
    H = np.array([
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    print(skeletonizing(H))