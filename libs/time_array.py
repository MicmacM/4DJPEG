"""
The goal of this script is to take a video, and extract it's pixels along the time t, so that a pixel gets 
represented by a time series.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def video_to_frames_array(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frames = []
    
    while True:
        ret, frame = cap.read()
        
        # If no frame is returned, the video has ended
        if not ret:
            break
        
        # Converting from opencv convention (BGR) to image convention (RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    #Creation of the time series
    frames_array = np.stack(frames, axis=-1) 
    
    return frames_array


if __name__ == "__main__":
    video_path = '../cresson.mp4'
    frames_array = video_to_frames_array(video_path)

    if frames_array is not None:
        print(f"Shape of frames array: {frames_array.shape}")
        #This shows the time series of the pixel (500,500)
        plt.imshow(frames_array[500,500,:,:])
        plt.show()
