from typing import List
import torch
import numpy as np
import dlib
import cv2
# from mmpose.apis import MMPoseInferencer
import face_recognition
import numpy as np
import fastapi
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()
detector = dlib.get_frontal_face_detector()

# @app.post("/face_keypoints")
# async def face_keypoints_batch(files: List[UploadFile]):
#     try:
#         keypoints = []
#         faces = []
#         data = []
#         for file in files:
#             content = await file.read()
#             np_array = np.frombuffer(content, np.uint8)
#             image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)


#             face_landmarks_list = face_recognition.face_landmarks(image)
#             if face_landmarks_list:
#                 # Extract lip keypoints (outer and inner)
#                 lip_points_outer = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
#                 lip_points_inner = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
#                 keypoints = lip_points_outer + lip_points_inner
#                 # Get the centroid of the lip
#                 centroid = (int(np.mean([point[0] for point in keypoints])),
#                             int(np.mean([point[1] for point in keypoints]))) 
#             data.append({"keypoints": keypoints, "centroid": centroid})
#             # face_landmarks_list = face_recognition.face_landmarks(image)
#             # face_locations = face_recognition.face_locations(image)
#             # print()
#             # print(face_landmarks_list, len(face_landmarks_list[0]["bottom_lip"]), len(face_landmarks_list[0]["top_lip"]))
#             # keypoints.append(face_landmarks_list)
#         print(data)
#         return data
#     except Exception as e:
#         return JSONResponse(content={"error": f"Error processing the image: {str(e)}"}, status_code=500)
    

def get_face_position(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces:
        return [[face.left(), face.top(), face.right(), face.bottom()] for face in faces]
    return []


@app.post("/face_keypoints/")
async def face_keypoints(file: UploadFile):
    try:
        data = {}
        content = await file.read()
        np_array = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        face_landmarks_list = face_recognition.face_landmarks(image)
        face_box = get_face_position(image)
        if face_landmarks_list:
            # Extract lip keypoints (outer and inner)
            lip_points_outer = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
            lip_points_inner = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
            keypoints = lip_points_outer + lip_points_inner
            # Get the centroid of the lip
            centroid = (int(np.mean([point[0] for point in keypoints])),
                        int(np.mean([point[1] for point in keypoints])))
            data["keypoints"] = keypoints
            data["centroid"] = centroid
        if face_box:
            data["face_position"] = face_box
        return data
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing the image: {str(e)}"}, status_code=500)