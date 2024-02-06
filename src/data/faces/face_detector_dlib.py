import cv2
import dlib
import numpy as np
import os

# Inicializar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Cargar el video
video_path = r'C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\data\faces\ingles\Homoeopathy in Female Problems by Dr  Meghna Shah_trimmed.mp4'
cap = cv2.VideoCapture(video_path)

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

# Reescalar todas las imágenes al tamaño máximo
for i in range(len(face_frames)):
    face_frames[i] = cv2.resize(face_frames[i], (max_width, max_height))

face_frames_np = np.array(face_frames)

print(type(face_frames_np))

# Guardar el array en un archivo npz
np.savez('face_frames.npz', face_frames=face_frames_np)

# Opcional: Mostrar el número total de caras recortadas
print(f"Total de caras recortadas guardadas: {len(face_frames)}")

# Guardar las imágenes como archivos individuales
for i in range(len(face_frames)):
    cv2.imwrite(os.path.join(f'face_{i+1}.jpg'), face_frames[i])

# Recuerda liberar el capturador de video
cap.release()


# Cargar el archivo npz
#data = np.load(r'C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\data\faces\face_frames.npz')

# Verificar las claves (keys) del archivo npz
#print("Claves del archivo npz:", data.files)

# Acceder al array que contiene los frames de las caras
#face_frames_np = data['face_frames']

# Verificar la forma (shape) del array
#print("Forma del array de frames de las caras:", face_frames_np.shape)

# Verificar el tipo de datos (dtype) del array
#print("Tipo de datos del array de frames de las caras:", face_frames_np.dtype)

# Visualizar el primer frame (por ejemplo)
#first_frame = face_frames_np[0]

# Mostrar el primer frame
#cv2.imwrite(os.path.join(f'test_npz.jpg'), first_frame)

