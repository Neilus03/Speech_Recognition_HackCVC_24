import whisper
import os
from tqdm import tqdm
import random

model = whisper.load_model("base")

directori = "/data3fast/users/group02/videos/tracks/"

files = list(os.walk(directori))
random.shuffle(files)
for carpeta_actual, carpetes, fitxers in tqdm(files):
    for fitxer in fitxers:
        ruta_completa = os.path.join(carpeta_actual, fitxer)
        if ruta_completa.endswith('.mp3'):
            result = model.transcribe(ruta_completa)
            with open(os.path.join(carpeta_actual, 'transcription.txt'), 'w') as file:
                file.write(result["text"])
                

'''
#In english below:   

model = whisper.load_model("base")

directory = "/data3fast/users/group02/videos/tracks/"
files = list(os.walk(directory))

random.shuffle(files)
for folder, subfolders, files in tqdm(files):
    for file in files:
        full_path = os.path.join(folder, file)
        if full_path.endswith('.mp3'):
            result = model.transcribe(full_path)
            with open(os.path.join(folder, 'transcription.txt'), 'w') as file:
                file.write(result["text"])'''