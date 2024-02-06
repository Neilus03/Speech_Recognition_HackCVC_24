import whisper
import os
from tqdm import tqdm
import random

model = whisper.load_model("base")

#result = model.transcribe("/data3fast/users/group02/videos/The Fear of Success Dr Marty Hauff at TEDxYouth@WISS_trimmed.mp3")
#print(result["text"])

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


