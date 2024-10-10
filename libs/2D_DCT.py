from math import cos, pi, sqrt
import numpy as np
from scipy.fftpack import dct
import time

def DCT_1D(x):
    return [sum([2*x[n]*cos(pi/len(x)*(n + 1/2)*k) for n in range(len(x))]) for k in range(len(x))]

def DCT_2D(x):
    N = len(x)
    X = np.zeros((N, N))

    # Precomputing cos for improved speed
    cos_matrix_1 = np.array([[cos(pi / N * (n1 + 0.5) * k1) for n1 in range(N)] for k1 in range(N)])
    cos_matrix_2 = np.array([[cos(pi / N * (n2 + 0.5) * k2) for n2 in range(N)] for k2 in range(N)])

    for k1 in range(N):
        for k2 in range(N):
            Xk1k2 = 0
            for n1 in range(N):
                for n2 in range(N):
                    Xk1k2 += x[n1, n2] * cos_matrix_1[k1, n1] * cos_matrix_2[k2, n2]
            
            # Normalization
            alpha_k1 = sqrt(1 / N) if k1 == 0 else sqrt(2 / N)
            alpha_k2 = sqrt(1 / N) if k2 == 0 else sqrt(2 / N)
            X[k1, k2] = alpha_k1 * alpha_k2 * Xk1k2

    return X

def DCT_2D_scipy(x):
    return dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")

if __name__ == "__main__":
    n = 8
    x = np.random.randint(0,10,size=(n,n))
    t0 = time.time()
    dct_result_manual = DCT_2D(x)
    t1 = time.time()
    dct_result_scipy = DCT_2D_scipy(x)
    t2 = time.time()
    print("DCT (wikipedia formula) : ", dct_result_manual)
    print("DCT (scipy) : ", dct_result_scipy)
    print("DCT manual exec time :", t1 - t0)
    print("DCT scipy exec time :", t2 - t1)

    # Well scipy is 100-1000x faster, what a beautiful optimisation...