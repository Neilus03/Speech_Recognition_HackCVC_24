import numpy as np
import dlib
import cv2
import face_recognition
import os

#base_dir = "/home/adriangar8/Documents/academia/CVC/hack_repo/src/data/faces"
#base_dir = r"C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\data\faces"
base_dir = '/data3fast/users/group02/videos/tracks'

# Iterar a través de los directorios y archivos
for root, dirs, files in os.walk(base_dir):

    for dir in dirs:
    
        npz_file_path = os.path.join(root, dir, 'face_frames.npz')
    
        if os.path.exists(npz_file_path):
    
            data = np.load(npz_file_path)

            frames = data['face_frames']

            # Find face landmarks
            face_landmarks_list = face_recognition.face_landmarks(frames[2])

            # Extract lip keypoints (outer and inner)
            lip_points_outer = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
            lip_points_inner = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']

            keypoints = list()

            centroid_image = frames[2].copy()

            # Optionally, visualize the keypoints on the image
            for point in lip_points_outer + lip_points_inner:

                # Save keypoints to the list
                keypoints.append(point)

                cv2.circle(frames[2], point, 1, (0, 255, 0), -1)

            # Get the centroid of the lip
            centroid = (int(np.mean([point[0] for point in lip_points_outer + lip_points_inner])),
                        int(np.mean([point[1] for point in lip_points_outer + lip_points_inner])))

            cv2.circle(centroid_image, centroid, 1, (0, 255, 0), -1)

            # Calculate distances from all keypoints to the centroid
            distances = [np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2) for point in keypoints]
            
            distances_np = np.array(distances)

            npz_file_path = os.path.join(root, dir, 'lip_keypoints.npz')
            print("npz_file_path:", npz_file_path)
            np.savez(npz_file_path, face_frames=distances_np)

print("Done!")