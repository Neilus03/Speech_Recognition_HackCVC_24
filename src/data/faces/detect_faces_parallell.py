import cv2
from tqdm import tqdm
import os
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
import random

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

def yolo_detect_faces(frame, frame_idx, data_parallel_list_mp, model):

    output = model(Image.fromarray(frame))
    results = Detections.from_ultralytics(output[0])

    x,y,w,h = 0, 0, 5, 5
    for result in results.xyxy:
        result = [int(r) for r in result]
        x1, y1, x2, y2 = result
        w_, h_ = x2 - x1, y2 - y1
        if w_*h_ > w*h:
            x,y,w,h = x1, y1, w_, h_

    coords = x,y,w,h
    retallat = frame[coords[0]:coords[0]+coords[-2],
                                       coords[1]:coords[1] + coords[-1]]

    try:
        data_parallel_list_mp[frame_idx] = cv2.resize(retallat, (128, 128))

    except: data_parallel_list_mp[frame_idx] = np.zeros((128, 128, 3))


def detect_face(frame, frame_idx, data_parallel_list_mp):
    # Carrega la cascada de classificadors Haar per a la detecció facial
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Converteix la imatge a escala de grisos
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta cares a la imatge
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Dibuixa rectangles al voltant de les cares detectades
    coords = 0, 0, 5, 5
    for (x, y, w, h) in faces:
        if w * h >= coords[-1] * coords[-2]:
            coords = x,y,w,h

    retallat = frame[coords[0]:coords[0]+coords[-2],
                                       coords[1]:coords[1] + coords[-1]]

    try:
        data_parallel_list_mp[frame_idx] = cv2.resize(retallat, (128, 128))

    except: data_parallel_list_mp[frame_idx] = np.zeros((128, 128, 3))
def get_all_videos(base_path):
    return sum([[(root, file) for file in files if file.endswith('.mp4')]
            for root, dirs, files in os.walk(base_path)], start = [])


def get_all_frames_from_video(video_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Initialize an empty list to store frames
    frames = []

    # Read frames until there are no more frames in the video
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Append the frame to the list
        frames.append(frame)

    # Release the VideoCapture object
    cap.release()

    return frames

def save_to_numpy(path, list_of_faces):
    np.savez(os.path.join(path, 'faces.npz'), list_of_faces)

if __name__ == '__main__':
    all_videos = random.shuffle(get_all_videos('/data3fast/users/group02/videos/tracks'))
    multiprocessing.set_start_method('spawn')
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    # load model
    model = YOLO(model_path).to('cuda')

    for path, filename in tqdm(all_videos):
        frames = get_all_frames_from_video(os.path.join(path, filename))
        # Initialize Manager and shared list for multiprocessing
        all_faces = [None for _ in frames]
        for idx, faces in enumerate(frames):
            yolo_detect_faces(faces, idx, all_faces, model)





        save_to_numpy(path, np.stack(all_faces))
