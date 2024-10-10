from math import cos, pi
import numpy as np
from scipy.fftpack import dct

def DCT_1D(x):
    return [sum([2*x[n]*cos(pi/len(x)*(n + 1/2)*k) for n in range(len(x))]) for k in range(len(x))]

def DCT_2D(x):
    X = []
    N = len(x)
    for k1 in range(N):
        for k2 in range(N):
            Xk1k2 = 0
            for n1 in range(N):
                c1 = cos(pi/N*(n1 + 1/2)*k1)
                for n2 in range(N):
                    c2 = cos(pi/N*(n2 + 1/2)*k2)
                    Xk1k2 += x[n1,n2]*c1*c2*2
n = 16
l = [4*i for i in range(n)]

print(DCT_1D(l))

dct_result = dct(l, type=2)

print(dct_result)

