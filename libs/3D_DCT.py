from math import cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fftpack import dct, idct
import time_array
import cv2

def clamp(color):
    return max(min(color, ))

def DCT_3D(x):
    x_r, x_g, x_b = x[:,:,0,:], x[:,:,1,:], x[:,:,2,:]
    dct_r = dct(dct(dct(x_r, axis=0, norm="ortho"), axis=1, norm="ortho"), axis=2, norm="ortho")
    dct_g = dct(dct(dct(x_g, axis=0, norm="ortho"), axis=1, norm="ortho"), axis=2, norm="ortho")
    dct_b = dct(dct(dct(x_b, axis=0, norm="ortho"), axis=1, norm="ortho"), axis=2, norm="ortho")
    return np.stack([dct_r, dct_g, dct_b], axis=-2) 

def IDCT_3D(x):
    x_r, x_g, x_b = x[:,:,0,:], x[:,:,1,:], x[:,:,2,:]
    idct_r = idct(idct(idct(x_r, axis=0, norm="ortho"), axis=1, norm="ortho"), axis=2, norm="ortho")
    idct_g = idct(idct(idct(x_g, axis=0, norm="ortho"), axis=1, norm="ortho"), axis=2, norm="ortho")
    idct_b = idct(idct(idct(x_b, axis=0, norm="ortho"), axis=1, norm="ortho"), axis=2, norm="ortho")
    return np.clip(np.stack([idct_r, idct_g, idct_b], axis=-2), 0, 255)

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

def norm3manhattan(x,y,z):
    return x+y+z

def manhattan_zigzag_matrix():
    zz = np.fromfunction(norm3manhattan, (8, 8, 8))
    pos = 0
    mini = 0
    tovisit = [(i,j,k) for i in range(8) for j in range(8) for k in range(8)]
    place_in_visit = 0
    while tovisit:
        if place_in_visit == len(tovisit):
            place_in_visit = 0
            mini += 1
        e = tovisit[place_in_visit]
        if zz[e] == mini:
            zz[e] = pos
            pos += 1
            tovisit.pop(place_in_visit)
            place_in_visit = 0
        else:
            place_in_visit += 1
    return zz


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


def quantize(im_array, quantization_matrix_3D):
    dct_arr = DCT_3D(im_array)
    quantize_rgbt = np.repeat(quantization_matrix_3D[:, :, np.newaxis, :], 3, axis=2)
    return dct_arr / quantize_rgbt

def unquantize(quantized_arr, quantization_matrix_3D):
    quantize_rgbt = np.repeat(quantization_matrix_3D[:, :, np.newaxis, :], 3, axis=2)
    unquantized_arr = quantized_arr * quantize_rgbt
    return IDCT_3D(unquantized_arr)


def visualize_frames_as_video(frames_array, interval=500):
    fig, ax = plt.subplots()
    frames_array = frames_array.astype(np.uint8)
    frame_display = ax.imshow(cv2.cvtColor(frames_array[..., 0], cv2.COLOR_BGR2RGB))

    def update_frame(t):
        """
        Update the frame for the animation.
        """

        frame_display.set_data(cv2.cvtColor(frames_array[..., t], cv2.COLOR_BGR2RGB))
        return frame_display,
    ani = animation.FuncAnimation(fig, update_frame, frames=frames_array.shape[-1], interval=interval, blit=True)

    plt.show()

def zigzag_encoding(matrix):
    zz = manhattan_zigzag_matrix()
    encoded = np.zeros(matrix.size)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                encoded[zz[i,j,k]] = matrix[i,j,k]

def delta_encoding(l):
    return [l[i] if i == 0 else l[i] - l[i-1] for i in range(len(l))]

def run_length_encoding(l):
    res = []
    nb_elem = 1
    for i in range(len(l)-1):
        if l[i+1] == l[i]:
            nb_elem += 1
        else:
            res.append((l[i], nb_elem))
            nb_elem = 1
    return res

#! NEED TO ADD THE LAST TUPLE
print(run_length_encoding([3,3,3,0,0,0,0,4,0,5,0,0,0,2,0,0]))

def matrix_size_in_bits(matrix):
    # Get size in bytes
    size_in_bytes = matrix.nbytes
    
    # Convert to bits
    size_in_bits = size_in_bytes * 8
    
    return size_in_bits



if __name__ == "__main__":
    video_path = "../cresson.mp4"
    base_x = 800
    base_y = 350
    increment = 12
    test_array = time_array.video_to_frames_array(video_path)
    reconstitued = np.zeros((increment*8, increment*8, 3, 8)).astype(int)
    echantillon = test_array[base_x:base_x + increment*8, base_y:base_y + increment*8, :, 2:10]
    for i in range(increment):
        for j in range(increment):
            sample = test_array[base_x + i*8:base_x + 8*(i+1), base_y + 8*j:base_y + 8*(j+1), :, 2:10]
            compressed = quantize(sample, Q)
            decompressed = unquantize(compressed, Q)
            reconstitued[8*i:8*(i+1), 8*j:8*(j+1), :, :] = decompressed
    #visualize_frames_as_video(echantillon, interval=500)
    #visualize_frames_as_video(reconstitued, interval=500)
    print("Compression level : ", (1- matrix_size_in_bits(compressed)/matrix_size_in_bits(echantillon))*100, "%")