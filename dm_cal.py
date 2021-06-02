import numpy as np
import math

def get_KNN(Matrix, K):
    ReMatrix = np.zeros(Matrix.shape)
    dimensional = Matrix.shape[0]
    Matrix_self = Matrix - np.eye(dimensional)
    for i in range(dimensional):
        a = [(Matrix_self[i,j], j)for j in range(dimensional)]
        a.sort(reverse=True)
        for j in range(K):
            e = int(a[j][1])
            ReMatrix[i, e] = 1
    return np.multiply(ReMatrix, Matrix)


def getSimilarMatrix(IP, γ_):
    dimensional = IP.shape[0]
    sd = np.zeros(dimensional)
    K = np.zeros((dimensional, dimensional))
    for i in range(dimensional):
        sd[i] = np.linalg.norm(IP[i]) ** 2
    gamad =  γ_*dimensional / np.sum(sd.transpose())
    for i in range(dimensional):
        for j in range(dimensional):
            K[i][j] = math.exp(-gamad * (np.linalg.norm(IP[i] - IP[j])) ** 2)
    return K


def getIsolateElement(A):
    return (np.sum(A, axis=1) == 0)

def row_norm(M):
    n1,n2 = M.shape
    result = M.copy()
    for i in range(n1):
        div = np.sum(M[i])
        if div==0:
            div = 1
        for j in range(n2):
            result[i][j] = M[i][j]/div
    return result




