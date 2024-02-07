#This file will take an image of lips and a graph and will overlap the graph on the image
# Function to overlay a networkx graph on an image

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2

def convert_bgr_to_grayscale(image_array):
    return np.dot(image_array[..., :3], [0.114, 0.587, 0.299])

def crop_image(image_array, positions, margin=15):
    x_values, y_values = zip(*positions.values())
    min_x, max_x = max(min(x_values) - margin, 0), min(max(x_values) + margin, image_array.shape[1])
    min_y, max_y = max(min(y_values) - margin, 0), min(max(y_values) + margin, image_array.shape[0])
    return image_array[int(min_y):int(max_y), int(min_x):int(max_x)], min_x, min_y

def overlay_graph_on_image(image_array, G, margin=15):
    grayscale_image = convert_bgr_to_grayscale(image_array)
    grayscale_image_bgr = cv2.cvtColor(grayscale_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #grayscale_image = np.stack([grayscale_image]*3).transpose(1,2,0)
    positions = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    cropped_image, min_x, min_y = crop_image(grayscale_image_bgr, positions, margin)
    adjusted_positions = {node: (pos[0] - min_x, pos[1] - min_y) for node, pos in positions.items()}
    
    # Convertir la imagen recortada a un formato que OpenCV pueda utilizar (debe estar en float32)
    cropped_image_cv2 = np.float32(cropped_image)
    
    # Crear una imagen en blanco del mismo tamaño que la imagen recortada para dibujar el grafo
    graph_image = np.zeros_like(cropped_image_cv2)
    # Dibujar los nodos y aristas del grafo en la imagen
    for node, (x, y) in adjusted_positions.items():
        cv2.circle(graph_image, (int(x), int(y)), 7, (190, 0, 70), -1)  # Dibujar nodo

    for edge in G.edges():
        pt1 = (int(adjusted_positions[edge[0]][0]), int(adjusted_positions[edge[0]][1]))
        pt2 = (int(adjusted_positions[edge[1]][0]), int(adjusted_positions[edge[1]][1]))
        cv2.line(graph_image, pt1, pt2, (255, 255, 255), 1)  # Dibujar arista
    
    # Combinar la imagen del grafo con la imagen recortada en escala de grises
    result_image = cv2.addWeighted(cropped_image_cv2, .5, graph_image, .5, 0)
    
    return result_image


if __name__ == '__main__':
    # Example usage with a mock image array and graph
    # Create a mock image array (e.g., 768x768 pixels)
    #make the mock image a random image
    mock_image_array = np.random.rand(768, 768, 3)*255
    # fuiste  tu puto mierda
    # Create a graph with 48 nodes and mock coordinates
    G_example = nx.Graph()
    for i in range(48):
        # Mock coordinates, replace with actual coordinates from your data
        G_example.add_node(i, x=np.random.randint(100, 668), y=np.random.randint(100, 668))

    # Add edges between each adjacent node (for visualization purposes)
    for i in range(47):
        G_example.add_edge(i, i+1)

    resulting_image_array = overlay_graph_on_image(mock_image_array, G_example)


    plt.figure(figsize=(10, 10))  # Puedes ajustar el tamaño de la figura a tu preferencia
    plt.imshow(resulting_image_array.astype(np.uint8), cmap='gray')  # Asegúrate de usar cmap='gray' si es una imagen en escala de grises
    plt.axis('off')  # Omitir los ejes para una visualización más limpia
    plt.show()