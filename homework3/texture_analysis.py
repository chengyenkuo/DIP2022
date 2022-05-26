import numpy as np
import basics
import random
import cv2

Lmasks=[
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    [[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]],
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
    [[1, 2, -1], [0, 0, 0], [1, -2, 1]],
    [[1, -2, -1], [2, 4, 2], [-1, -2, -1]],
    [[-1, 0, 1], [2, 0, -2], [-1, 0, 1]],
    [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
]
Ldiv=[36, 12, 12, 12, 4, 4, 12, 4, 4]

vL=np.array([1, 4, 6, 4, 1])     # L(level) : Calculate the symmetric weighted local average
vB=np.array([-1, -2, 0, 2, 1])    # B(border): Identify borders
vS=np.array([-1, 0, 2, 0, -1])   # S(spots): Identify spots
vR=np.array([1, -4, 6, -4, 1])   # R(ripples) : Identify the image as rippled
################################################

def rho(cur, μ):  # ρ
    return np.mean(np.absolute(cur-μ))

def norm(source):
    #print('max=', np.max(source), ' min=', np.min(source))
    return (source-np.min(source))*(255/(np.max(source)-np.min(source)))

def Laws(source):
    (R, C), W=source.shape, 15
    
    # 1. eliminating the influence of light intensity
    pad, elim=basics.padding(source, W), np.copy(source)
    for r in range(R):
        for c in range(C):
            elim[r][c]-=np.mean(pad[r:r+W, c:c+W])
    #cv2.imwrite('elim.png', elim)
    
    
    # 2. masks
    LBSR, masks=np.array([vL, vB, vS, vR]), np.zeros((4, 4, 5, 5))
    for i in range(4):
        for j in range(4):
            for r in range(5):
                for c in range(5):
                    masks[i][j][r][c]=LBSR[i][r]*LBSR[j][c]
    
    for i in range(4):
        for j in range(4):
            print(masks[i][j])
    
    
    # 3. filtering
    pad, M=basics.padding(elim, 5), np.zeros((4, 4, R, C))
    for i in range(4):
        for j in range(4):
            for r in range(R):
                for c in range(C):
                    M[i][j][r][c]=np.sum(pad[r:r+5, c:c+5]*masks[i][j])
            print('LBSR_M'+str(i)+str(j))
            #cv2.imwrite('LBSR_M'+str(i)+str(j)+'.png', norm(M[i][j]))
    
    # 4. texture maps
    T, W=np.zeros((4, 4, R, C)), 15
    for i in range(4):
        for j in range(4):
            pad=basics.padding(M[i][j], W)
            for r in range(R):
                for c in range(C):
                    T[i][j][r][c]=np.sum(np.absolute(pad[r:r+W, c:c+W]))
            print('LBSR_T'+str(i)+str(j))
            #cv2.imwrite('LBSR_T'+str(i)+str(j)+'.png', norm(T[i][j]))
    
    # 5. energy maps
    E=np.zeros((9, R, C))
    for k in range(3):
        E[k]=T[k+1][k+1]
        #cv2.imwrite('LBSR_E'+str(k)+'.png', norm(E[k]))
    k=3
    for i in range(4):
        for j in range(i+1, 4):
            E[k]=(T[i][j] + T[j][i])/2
            #cv2.imwrite('LBSR_E'+str(k)+'.png', norm(E[k]))
            k+=1
    return E

def k_means(K, E):
    (R, C, D)=E.shape
    
    μs=np.zeros((K, D))    # μ
    for k in range(K):
        μs[k]=E[random.randint(0,R)][random.randint(0,C)]
    
    cnt, out=0, np.zeros((R,C)).astype(int)
    while True:
        clusters = [set() for _ in range(K)]
        for r in range(R):
            for c in range(C):
                rhos=np.zeros(K)
                for k in range(K):
                    rhos[k]=rho(E[r][c], μs[k])
                out[r][c]=np.argmin(rhos)
                clusters[out[r][c]].add(tuple(E[r][c]))
        for k in range(K):
            n, μs[k]=len(clusters[k]), np.zeros(D)
            print(n)
            for e in clusters[k]:
                for i in range(D):
                   μs[k][i]+=e[i]/n
            print(μs[k])
        kmeans=out*(255/K)
        #cv2.imwrite('kmeans4_'+str(cnt)+'.png', kmeans)
        
        cnt+=1
        if (cnt > 20):
            break
    return out

if __name__ == '__main__':
    # (a)
    Lmasks = np.array(Lmasks).astype(float) # preprocess
    for k in range(9):
        Lmasks[k] /= Ldiv[k]
    
    sample2=cv2.imread('./SampleImage/sample2.png', cv2.IMREAD_GRAYSCALE)
    np.save('energy_maps', Laws(sample2))
    E, K=np.swapaxes(np.load('energy_maps.npy'), 0, 1), 4
    result6=k_means(K, np.swapaxes(E, 1, 2))
    cv2.imwrite('result6.png', result6)
    