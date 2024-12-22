from math import cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fftpack import dct, idct
import time_array
import cv2
from tqdm import tqdm

def clamp(color):
    return max(min(color, 255), 0)

def DCT_2D(x):
    x_r, x_g, x_b = x[:,:,0], x[:,:,1], x[:,:,2]
    dct_r = dct(dct(x_r, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_g = dct(dct(x_g, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_b = dct(dct(x_b, axis=0, norm="ortho"), axis=1, norm="ortho")
    return np.stack([dct_r, dct_g, dct_b], axis=-1) 

def IDCT_2D(x):
    x_r, x_g, x_b = x[:,:,0], x[:,:,1], x[:,:,2]
    idct_r = idct(idct(x_r, axis=0, norm="ortho"), axis=1, norm="ortho")
    idct_g = idct(idct(x_g, axis=0, norm="ortho"), axis=1, norm="ortho")
    idct_b = idct(idct(x_b, axis=0, norm="ortho"), axis=1, norm="ortho")
    return np.clip(np.stack([idct_r, idct_g, idct_b], axis=-1), 0, 255)

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


def quantize(im_array, quantization_matrix_2D):
    dct_arr = DCT_2D(im_array)
    Q = np.repeat(quantization_matrix_2D[:, :, np.newaxis], 3, axis=2)
    return (dct_arr / Q).astype(int)

def unquantize(quantized_arr, quantization_matrix_2D):
    quantize_rgbt = np.repeat(quantization_matrix_2D[:, :, np.newaxis], 3, axis=2)
    unquantized_arr = quantized_arr * quantize_rgbt
    return IDCT_2D(unquantized_arr).astype(int)


def zigzag_encoding(block):
    zigzag_indices = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ])
    encoded_r = [0] * 64
    encoded_g = [0] * 64
    encoded_b = [0] * 64
    for i in range(8):
        for j in range(8):
            encoded_r[zigzag_indices[i,j]] = block[i, j, 0]
            encoded_g[zigzag_indices[i,j]] = block[i, j, 1]
            encoded_b[zigzag_indices[i,j]] = block[i, j, 2]
    return encoded_r, encoded_g, encoded_b


def zigzag_decoding(encoded_r, encoded_g, encoded_b):
    zigzag_indices = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ])
    block = np.zeros((8, 8, 3))
    for i in range(8):
        for j in range(8):
            block[i,j,0] = encoded_r[zigzag_indices[i,j]]
            block[i,j,1] = encoded_g[zigzag_indices[i,j]]
            block[i,j,2] = encoded_b[zigzag_indices[i,j]]
    return block

def delta_encoding(l):
    return [l[i] if i == 0 else l[i] - l[i-1] for i in range(len(l))]

def delta_decoding(l):
    base = l[0]
    res = [base]
    for i in range(1,len(l)):
        res.append(res[-1] + l[i])
    return res

def run_length_encoding(l):
    res = []
    nb_elem = 1
    for i in range(len(l)-1):
        if l[i+1] == l[i]:
            nb_elem += 1
        else:
            res.append((l[i], nb_elem))
            nb_elem = 1
    res.append((l[-1], nb_elem))
    return res

def run_length_decoding(l):
    res = []
    for (e,n) in l:
        res += [e]*n
    return res

def process_frame(frame, Q):
    h, w, _ = frame.shape
    processed_frame = np.zeros_like(frame, dtype=int)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            for c in range(3):  # Process each color channel
                block = frame[i:i+8, j:j+8, c]
                dct_block = DCT_2D(block)
                quantized_block = quantize(dct_block, Q)
                dequantized_block = dequantize(quantized_block, Q)
                idct_block = IDCT_2D(dequantized_block)
                processed_frame[i:i+8, j:j+8, c] = np.clip(idct_block, 0, 255)
    return processed_frame


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

def matrix_size_in_bits(matrix):
    # Get size in bytes
    size_in_bytes = matrix.nbytes
    
    # Convert to bits
    size_in_bits = size_in_bytes * 8
    
    return size_in_bits

def number_of_bits_ac(lst):
    total_bits = 0
    for compressed_8x8_array in lst:
        for chanel_arrays in compressed_8x8_array:
            for tup in chanel_arrays:
                for num in tup:
                    total_bits += int(num).bit_length()  # Sum the bit length of each integer
    return total_bits

def number_of_bits_dc(lst):
    total_bits = 0
    for dc in lst:
        for comp in dc:
            total_bits += int(comp).bit_length()  # Sum the bit length of each integer
    return total_bits


if __name__ == "__main__":
    video_path = "../cresson.mp4"
    base_x = 800
    base_y = 350
    #Number of spatial chunk of size 8
    spatial_i = 12
    t0 = 2
    #Number of temporal chunk of size 8
    temporal_i = 10
    test_array = time_array.video_to_frames_array(video_path)
    compressed_data = np.zeros((spatial_i*8, spatial_i*8, 3, temporal_i*8)).astype(int)
    reconstitued = np.zeros((spatial_i*8, spatial_i*8, 3, temporal_i*8)).astype(int)
    echantillon = test_array[base_x:base_x + spatial_i*8, base_y:base_y + spatial_i*8, :, t0:t0+(temporal_i+1)*8]
    dc_values = []
    ac_values = []
    for t in tqdm(range(t0, t0 + temporal_i*8), desc="t"):
        for i in range(spatial_i):
            for j in range(spatial_i):
                sample = test_array[base_x + i*8:base_x + 8*(i+1), base_y + 8*j:base_y + 8*(j+1), :, t]

                compressed = quantize(sample, quantization_matrix)

                ac = [compressed[x,y,:] for x in range(8) for y in range(8)]
                dc = ac.pop(0)

                zz_compressed_r, zz_compressed_g, zz_compressed_b = zigzag_encoding(compressed)
                compressed_ac_r = run_length_encoding(zz_compressed_r)
                compressed_ac_g = run_length_encoding(zz_compressed_g)
                compressed_ac_b = run_length_encoding(zz_compressed_b)
                ac_values.append((compressed_ac_r, compressed_ac_b, compressed_ac_g))

                #! RUN LENGTH, DELTA AND ZIGZAG DECODING
                decompressed_ac_r = run_length_decoding(compressed_ac_r)
                decompressed_ac_g = run_length_decoding(compressed_ac_g)
                decompressed_ac_b = run_length_decoding(compressed_ac_b)

                decompressed_zz = zigzag_decoding(decompressed_ac_r, decompressed_ac_g, decompressed_ac_b)

                #! 3D IDCT AND DEQUANTIZATION
                decompressed = unquantize(decompressed_zz, quantization_matrix)
                reconstitued[8*i:8*(i+1), 8*j:8*(j+1), :, t-t0] = decompressed


    dc_values = delta_encoding(dc_values)
    #! EOF

    #! DEBUG ONLY
    print("number of bits ac", number_of_bits_ac(ac_values))
    print("number of bits dc", number_of_bits_dc(dc_values))
    #print(ac_values)
    print("matrix size in bit", matrix_size_in_bits(echantillon))
    visualize_frames_as_video(echantillon, interval=500)
    visualize_frames_as_video(reconstitued, interval=500)
    print("Compression level : ", (1- (number_of_bits_ac(ac_values)+number_of_bits_dc(dc_values))/matrix_size_in_bits(echantillon))*100, "%")