import dct_3D

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
    for i in range(spatial_i):
        for j in range(spatial_i):
            for t in range(temporal_i):
                sample = test_array[base_x + i*8:base_x + 8*(i+1), base_y + 8*j:base_y + 8*(j+1), :, t0+t*8:t0+(t+1)*8]
                
                #! 3D DCT COMPRESSION AND QUANTIZATION
                compressed = quantize(sample, Q)

                #! RUN LENGTH, DELTA AND ZIGZAG ENCODING
                ac = [compressed[x,y,:,z] for x in range(8) for y in range(8) for z in range(8)]
                dc = ac.pop(0)

                dc_values.append(dc)

                #? We take the dc value too, remove it in a later version perhaps ?
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
                decompressed = unquantize(decompressed_zz, Q)
                reconstitued[8*i:8*(i+1), 8*j:8*(j+1), :, 8*t:8*(t+1)] = decompressed
    
    dc_values = delta_encoding(dc_values)
    #! EOF

    #! DEBUG ONLY
    print(number_of_bits_ac(ac_values))
    print(number_of_bits_dc(dc_values))
    #print(ac_values)
    print(matrix_size_in_bits(echantillon))
    visualize_frames_as_video(echantillon, interval=500)
    visualize_frames_as_video(reconstitued, interval=500)
    print("Compression level : ", (1- (number_of_bits_ac(ac_values)+number_of_bits_dc(dc_values))/matrix_size_in_bits(echantillon))*100, "%")
