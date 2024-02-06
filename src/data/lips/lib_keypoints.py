import numpy as np
import dlib
import cv2
import face_recognition
# from mmpose.apis import MMPoseInferencer

data = np.load(r'C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\data\faces\face_frames.npz')
frames = data['face_frames']

# Convert the image to grayscale, as the detector expects a grayscale image
gray_image = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

# Find face landmarks
face_landmarks_list = face_recognition.face_landmarks(frames[0])

# Extract lip keypoints (outer and inner)
lip_points_outer = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
lip_points_inner = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']

# Optionally, visualize the keypoints on the image
for point in lip_points_outer + lip_points_inner:
    cv2.circle(frames[0], point, 1, (0, 255, 0), -1)

# Display the image with keypoints
cv2.imshow("Lip Keypoints", frames[0])
cv2.waitKey(0)
cv2.destroyAllWindows()