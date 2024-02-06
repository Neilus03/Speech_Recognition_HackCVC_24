import random

import cv2
import dlib
import numpy as np
import os
import face_recognition
import cv2

base_dir = r'C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\data\faces'

def get_all_videos(base_path):
    return sum([[(root, file) for file in files if file.endswith('.mp4')]
            for root, dirs, files in os.walk(base_path)], start = [])

all_videos = get_all_videos(base_dir)

# Iterar recursivamente en el directorio base
random.shuffle(all_videos)

for root, file in all_videos:
    if file.endswith('.mp4'):
        video_path = os.path.join(root, file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Cannot be open: {video_path}")
            continue

        frames = []
        max_width = 0
        max_height = 0

        while cap.isOpened():

            ret, frame = cap.read()

            face_landmarks_list = face_recognition.face_landmarks(frame)

            if face_landmarks_list:

                # Extract lip keypoints (outer and inner)
                lip_points_outer = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
                lip_points_inner = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']

                keypoints = lip_points_outer + lip_points_inner

                #for point in keypoints:
                    #cv2.circle(frame, point, 1, (0, 255, 0), thickness=-1)  # Dibuja un círculo verde para cada keypoint
                
                #cv2.imwrite(r'C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\data\faces\keypoints_frames.jpg', frame)
                            
                # Get the centroid of the lip
                centroid = (int(np.mean([point[0] for point in keypoints])),
                            int(np.mean([point[1] for point in keypoints])))

                # Calculate distances from all keypoints to the centroid
                distances = [np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2) for point in keypoints]

                max_dist = distances[47]

                print("dist[0]", distances[0])
                distances_norm = [dis/max_dist for dis in distances]
                print("dist_norm[0]", distances_norm[0])

            if not ret:
                break

        if not frames:
            print(f"No se detectaron labios en el video: {video_path}")
            cap.release()
            continue

        # Convertir a array de NumPy y guardar en archivo .npz
        face_frames_np = np.array(resized_face_frames)
        npz_file_path = os.path.join(root, 'frames.npz')
        np.savez(npz_file_path, face_frames=face_frames_np)

        print(f"Total de caras recortadas guardadas en {npz_file_path}: {len(resized_face_frames)}")
        cap.release()

