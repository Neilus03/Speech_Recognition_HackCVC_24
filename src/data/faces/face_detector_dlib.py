import cv2
import dlib
import numpy as np
import os

# Inicializar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

base_dir = '/data3fast/users/group02/videos/tracks'

# Iterar recursivamente en el directorio base
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Verificar si el archivo tiene la extensión .mp4
        if file.endswith('.mp4'):
            # Construir la ruta completa del archivo
            video_path = os.path.join(root, file)

            # Cargar el video
            cap = cv2.VideoCapture(video_path)

            # Verificar si el video se abrió correctamente
            if not cap.isOpened():
                print(f"No se pudo abrir el video: {video_path}")
                continue

            # Lista para almacenar los recortes de las caras
            face_frames = []

            max_width = 0
            max_height = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convertir el cuadro a escala de grises para mejorar la detección
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detectar rostros en el cuadro
                faces = detector(gray)

                # Verificar si se detectaron rostros antes de intentar recortarlos
                if len(faces) > 0:
                    # Recortar cada rostro detectado
                    for face in faces:
                        x, y = face.left(), face.top()
                        x1, y1 = face.right(), face.bottom()
                        # Asegurarse de que el recorte no salga del cuadro
                        x, y, x1, y1 = max(x, 0), max(y, 0), min(x1, frame.shape[1]), min(y1, frame.shape[0])
                        # Recortar el rostro
                        face_image = frame[y:y1, x:x1]

                        # Actualizar el tamaño máximo de la imagen
                        max_width = max(max_width, face_image.shape[1])
                        max_height = max(max_height, face_image.shape[0])

                        # Guardar el recorte en la lista
                        face_frames.append(face_image)

            # Verificar si se detectaron rostros
            if len(face_frames) == 0:
                print(f"No se detectaron rostros en el video: {video_path}")
                cap.release()
                continue

            # Reescalar todas las imágenes al tamaño máximo
            for i in range(len(face_frames)):
                face_frames[i] = cv2.resize(face_frames[i], (max_width, max_height))

            face_frames_np = np.array(face_frames)

            # Guardar el array en un archivo npz en la misma carpeta que el video
            npz_file_path = os.path.join(root, 'face_frames.npz')
            np.savez(npz_file_path, face_frames=face_frames_np)

            # Mostrar el número total de caras recortadas guardadas
            print(f"Total de caras recortadas guardadas en {npz_file_path}: {len(face_frames)}")

            # Liberar el capturador de video
            cap.release()
