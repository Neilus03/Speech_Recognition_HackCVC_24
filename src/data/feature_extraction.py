import random
import dlib
import numpy as np
import os
import face_recognition
import cv2
from tqdm import tqdm

base_dir = "/data3fast/users/group02/videos/tracks/"
# base_dir = "/home/adriangar8/Documents/academia/CVC/hack_repo/src/data/faces"

def get_all_videos(base_path):
    return sum([[(root, file) for file in files if file.endswith('_trimmed.mp4')]
            for root, dirs, files in os.walk(base_path)], start = []
            )

all_videos = get_all_videos(base_dir)

print()
print(len(all_videos))
print()

# Iterar recursivamente en el directorio base
random.shuffle(all_videos)

for root, file in tqdm(all_videos):
    if file.endswith('.mp4'):

        try:

            video_path = os.path.join(root, file)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Cannot be open: {video_path}")
                continue

            frames = []
            max_width = 0
            max_height = 0
            kps = []
            while cap.isOpened():
                
                ret, frame = cap.read()

                if not ret:
                    break

                frames.append(frame) # hello

                face_landmarks_list = face_recognition.face_landmarks(frame)

                if face_landmarks_list:

                    # Extract lip keypoints (outer and inner)
                    lip_points_outer = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
                    lip_points_inner = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']

                    keypoints = lip_points_outer + lip_points_inner
        
                    # Get the centroid of the lip
                    centroid = (int(np.mean([point[0] for point in keypoints])),
                                int(np.mean([point[1] for point in keypoints])))

                    # Calculate distances from all keypoints to the centroid
                    distances = [np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2) for point in keypoints]

                    max_dist = distances[47]

                    distances_norm = [dis/max_dist for dis in distances]

                    # Convertir a array de NumPy y guardar en archivo .npz
                    keypoints_np = np.array(keypoints).reshape(2 * 48)
                    distances_np = np.array(distances_norm).reshape(48)

                    # hstack = horizontal stack 
                    features_np = np.hstack((keypoints_np, distances_np))
                    kps.append(features_np)
                    # print(len(kps))
            npz_file_path = os.path.join(root, 'features_ok_finals_seqs.npz')
            np.savez(npz_file_path, face_frames=np.stack(kps))

            if not frames:

                print(f"Not video: {video_path}")
                cap.release()
                continue

            cap.release()

        except Exception as e:

            print(f"Error: {e}")
            continue
        
