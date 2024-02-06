import numpy as np
import cv2
import face_recognition
import os
from multiprocessing import Pool

def process_file(npz_file_path):
    if os.path.exists(npz_file_path):
        data = np.load(npz_file_path)
        frames = data['face_frames']

        # Find face landmarks
        face_landmarks_list = face_recognition.face_landmarks(frames[2])

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
            
            distances_np = np.array(distances)

            output_npz_file_path = os.path.join(os.path.dirname(npz_file_path), 'lip_keypoints.npz')
            np.savez(output_npz_file_path, lip_keypoints=distances_np)

            print(f"Processed: {output_npz_file_path}")

def main():
    base_dir = '/data3fast/users/group02/videos/tracks'
    npz_paths = []

    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            npz_file_path = os.path.join(root, dir, 'face_frames.npz')
            npz_paths.append(npz_file_path)

    # Utilizar multiprocessing para procesar los archivos
    pool = Pool(processes=os.cpu_count())  # Crea un pool de trabajadores igual al número de CPUs
    pool.map(process_file, npz_paths)

    pool.close()
    pool.join()

    print("Done!")

if __name__ == "__main__":
    main()
