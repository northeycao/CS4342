import numpy as np

A = np.array(([1,2,3,4,5], [6,7,8,9,10], [1,2,3,4,5], [6,7,8,9,10], [1,2,3,4,5]))
B = np.array([6,7,8,9,10])
C = np.array([10,11,12,13,14])

D = np.array(([1, -2], [3, 4]))
E = np.array([1, 1])

def problem1 (A, B, C):
    return np.dot(A, B) - C

# print(problem1(A,B,C))

def problem2 (A):
    rows = A.shape[0]
    return np.ones(rows)

# print(problem2(A))

def problem3 (A):
    for i in range(min(A.shape[0], A.shape[1])):
        A[i,i] = 0
    return A
# print(problem3(A))

def problem4 (A, i):
    return np.sum(A[i])
# print(problem4(A,1))

def problem5 (A, c, d):
    S = np.array([])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if (A[i,j] >= c and A[i,j] <= d):
                S = np.append(S, A[i,j])
    if (not S.size == 0):
        return np.mean(S)
    else:
        return 0;
# print(problem5(A, 1, 5))

def problem6 (A, k):
    n = A.shape[0]
    S = np.zeros((n, k))

    eigenvalues, eigenvectors = np.linalg.eig(A)

    for i in range(n):
        eigenvalues[i] = np.linalg.norm(eigenvalues[i])
    # print(eigenvalues)
    # print(eigenvectors)

    for i in range(k):
        maxind = eigenvalues.argmax()
        for j in range(n):
            if eigenvectors.ndim == 2:
                S[j,i] = eigenvectors[j, maxind]
            else:
                S[j,i] = eigenvalues[maxind]
        eigenvalues = np.delete(eigenvalues, maxind)
        eigenvectors = np.delete(eigenvectors, maxind, axis=1)
    return S
# print(problem6(A, 5))


def problem7 (A, x):
    return np.linalg.solve(A, x)
# print(problem7(D, E))

def problem8 (x, k):
    x = np.array([x])
    return np.repeat(x, k, axis=0)
# print(problem8(B, 3))

def problem9 (A):
    return np.random.permutation(A)
# print(problem9(A))

def problem10 (A):
    return np.mean(A, axis=1)
# print(problem10(A))

def problem11 (n, k):
    rng = np.random.default_rng()
    A = np.zeros(n)
    for i in range(n):
        A[i] = rng.integers(0, k, endpoint=True)
        if A[i]%2 == 0:
            A[i] = -1
    return A
# print(problem11(7, 20))

def problem12 (A, b):
    rows = A.shape[0]
    cols = A.shape[1]
    B = np.repeat(b, cols, axis=0)
    B = np.reshape(B, (rows, cols))
    return A + B
# print(problem12(A,B))

M = np.array([[[1, 1, 1],[1, 1, 1],[1, 1, 1]],
              [[2, 2, 2],[2, 2, 2],[2, 2, 2]],
              [[3, 3, 3],[3, 3, 3],[3, 3, 3]]])

def problem13 (A):
    n = A.shape[0]
    m = A.shape[1]
    B = np.zeros((n, m*m))
    for i in range(n):
        B[i] = A[i].reshape(-1)
    return B.transpose()
# print(problem13(M))
