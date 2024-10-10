from math import cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def clamp(color):
    return max(min(color, ))

def DCT_3D(x):
    return dct(dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho"), axis=2, norm="ortho")

quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def quantization_matrix_3D(quantization_matrix_2D):
    Q = np.zeros((8,8,8))
    for i in range(8):
        for j in range(8):
            for t in range(8):
                Q[i,j,t] = quantization_matrix_2D[i,j] + sqrt(i**2 + j**2 + t**2)*t
    return Q



Q = quantization_matrix_3D(quantization_matrix)

def plot_3d_scatter(matrix_3d):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.indices(matrix_3d.shape)
    values = matrix_3d.flatten()

    scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=values, cmap='viridis')
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    colorbar.set_label('Intensity')
    ax.set_xlabel('i axis')
    ax.set_ylabel('t axis')
    ax.set_zlabel('j axis')
    plt.show()


plot_3d_scatter(Q)

def quantize(arr, quantization_matrix_3D):
    dct_arr = DCT_3D(arr)
    return np.round(4*dct_arr/quantization_matrix_3D).astype(int)