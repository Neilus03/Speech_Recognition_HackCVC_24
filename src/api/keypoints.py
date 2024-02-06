from typing import List
import torch
import numpy as np
# import dlib
import cv2
# from mmpose.apis import MMPoseInferencer
import face_recognition
import fastapi
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()

@app.post("/face_keypoints/")
async def face_keypoints(files: List[UploadFile]):
    try:
        keypoints = []
        for file in files:
            content = await file.read()
            np_array = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            face_landmarks_list = face_recognition.face_landmarks(image)
            keypoints.append(face_landmarks_list)
        return keypoints
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing the image: {str(e)}"}, status_code=500)