import numpy as np
import dlib
import cv2
import face_recognition

data = np.load("/home/adriangar8/Documents/academia/CVC/hack_repo/src/data/faces/arabe/face_frames.npz")

frames = data['face_frames']

# Save the first frame to a file
cv2.imwrite("test.jpg", frames[2])

# Find face landmarks
face_landmarks_list = face_recognition.face_landmarks(frames[2])

# Extract lip keypoints (outer and inner)
lip_points_outer = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']
lip_points_inner = face_landmarks_list[0]['top_lip'] + face_landmarks_list[0]['bottom_lip']

keypoints = list()

centroid_image = frames[2].copy()

# Optionally, visualize the keypoints on the image
for point in lip_points_outer + lip_points_inner:

    # Save keypoints to the list
    keypoints.append(point)

    cv2.circle(frames[2], point, 1, (0, 255, 0), -1)

# Get the centroid of the lip
centroid = (int(np.mean([point[0] for point in lip_points_outer + lip_points_inner])),
            int(np.mean([point[1] for point in lip_points_outer + lip_points_inner])))

cv2.circle(centroid_image, centroid, 1, (0, 255, 0), -1)

"""
cv2.imshow("Lip centroid", centroid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Optionally, visualize the keypoints on the image and draw lines to the centroid
for point in lip_points_outer + lip_points_inner:
    # Save keypoints to the list
    keypoints.append(point)

    cv2.circle(frames[2], point, 1, (0, 255, 0), -1)
    
    # Draw a line from the point to the centroid
    cv2.line(frames[2], point, centroid, (0, 255, 0), 1)

# Calculate distances from all keypoints to the centroid
distances = [np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2) for point in keypoints]

print(distances)

# Display the image with keypoints
cv2.imshow("Lip Keypoints", frames[2])
cv2.waitKey(0)
cv2.destroyAllWindows()