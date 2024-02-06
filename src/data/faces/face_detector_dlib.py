import cv2
import dlib
import numpy as np

# Inicializar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Cargar el video
video_path = 'C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\data\faces\ingles\Homoeopathy in Female Problems by Dr  Meghna Shah_trimmed.mp4'
cap = cv2.VideoCapture(video_path)

# Lista para almacenar los recortes de las caras
face_frames = []

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
        # Guardar el recorte en la lista
        face_frames.append(face_image)

# Opcional: Mostrar el número total de caras recortadas
print(f"Total de caras recortadas guardadas: {len(face_frames)}")

# Aquí puedes procesar los recortes de las caras almacenados en face_frames
# Por ejemplo, mostrar el primer rostro recortado
if len(face_frames) > 0:
    cv2.imshow('Primer rostro recortado', face_frames[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Recuerda liberar el capturador de video
cap.release()
