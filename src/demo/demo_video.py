import os
import numpy as np
import face_recognition
import cv2
import requests
from io import BytesIO
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt


def create_first_graph(keypoints, centroid):
    G = nx.Graph()

    # Add nodes with attributes
    for i, keypoint in enumerate(keypoints):
        G.add_node(f"keypoint {i}", coordinates=keypoint)
    G.add_node("centroid", coordinates=centroid)  # Add centroid as a node

    # Add edges from each node to the centroid
    for i, keypoint in enumerate(keypoints):
        G.add_edge(f"keypoint {i}", "centroid")

    return G

def create_second_graph(keypoints, centroid):
    G = nx.Graph()

    # Add nodes with attributes
    for i, keypoint in enumerate(keypoints):
        G.add_node(f"keypoint {i}", coordinates=keypoint)

    # Add edges between consecutive keypoints forming a cyclic connection
    num_keypoints = len(keypoints)
    for i in range(num_keypoints - 1):
        G.add_edge(f"keypoint {i}", f"keypoint {i+1}")
    # Connect the last keypoint to the first one to form a cycle
    G.add_edge(f"keypoint {num_keypoints - 1}", f"keypoint 0")

    return G

def inference_video(frame):

    image_pil = Image.fromarray(frame)
    image_bytes_io = BytesIO()
    image_pil.save(image_bytes_io, format='PNG')

    # Call the API
    files = {'file': ('image.png', image_bytes_io.getvalue(), 'image/png')}
    response = requests.post("http://158.109.8.116:8000/face_keypoints/", files=files)

    if response.ok:
        # print("API called")
        data_dict = response.json()

    # Extract the keypoints, face_positions and the centroid of the dictionary
    keypoints = data_dict["keypoints"]
    face_positions = data_dict["face_position"][0]
    centroid = data_dict["centroid"]

    # Create and visualize the first graph
    first_graph = create_first_graph(keypoints, centroid)

    # Create and visualize the second graph
    second_graph = create_second_graph(keypoints, centroid)

    return first_graph, second_graph, face_positions
