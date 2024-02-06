import cv2
import dlib
import numpy as np
import os

# Inicializar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

base_dir = '/data3fast/users/group02/videos/tracks/n6ONKshWSMg'

# Iterar recursivamente en el directorio base
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"No se pudo abrir el video: {video_path}")
                continue

            face_frames = []
            max_width = 0
            max_height = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                if faces:
                    max_area = 0
                    max_face = None

                    for face in faces:
                        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
                        area = (x1 - x) * (y1 - y)
                        if area > max_area:
                            max_area = area
                            max_face = face

                    if max_face:
                        x, y, x1, y1 = max_face.left(), max_face.top(), max_face.right(), max_face.bottom()
                        x, y, x1, y1 = max(x, 0), max(y, 0), min(x1, frame.shape[1]), min(y1, frame.shape[0])
                        face_image = frame[y:y1, x:x1]
                        face_frames.append(face_image)

                        # Actualizar el tamaño máximo según sea necesario
                        max_width = max(max_width, x1 - x)
                        max_height = max(max_height, y1 - y)

            if not face_frames:
                print(f"No se detectaron rostros en el video: {video_path}")
                cap.release()
                continue

            # Reescalar todas las imágenes de caras al tamaño máximo encontrado
            resized_face_frames = [cv2.resize(face, (max_width, max_height)) for face in face_frames]

            # Convertir a array de NumPy y guardar en archivo .npz
            face_frames_np = np.array(resized_face_frames)
            npz_file_path = os.path.join(root, 'face_frames.npz')
            np.savez(npz_file_path, face_frames=face_frames_np)

            print(f"Total de caras recortadas guardadas en {npz_file_path}: {len(resized_face_frames)}")
            cap.release()

