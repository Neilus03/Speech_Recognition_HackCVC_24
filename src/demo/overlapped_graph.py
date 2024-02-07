#This file will take an image of lips and a graph and will overlap the graph on the image
# Function to overlay a networkx graph on an image

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def convert_bgr_to_grayscale(image_array):
    """
    Converts an RGB image to grayscale.
    Returns:
    - Grayscale image array.
    """
    return np.dot(image_array[..., :3], [0.114, 0.587, 0.299])
    #return np.mean(image_array, axis=-1)

def crop_image(image_array, positions, margin=15):
    """
    Crops the image to the bounding box of the graph positions with some margin.
    
    Parameters:
    - image_array: numpy array of the image
    - positions: dictionary of node positions
    - margin: integer margin to add around the bounding box
    
    Returns:
    - Cropped image array, min_x, min_y for adjusting graph node positions.
    """
    x_values, y_values = zip(*positions.values())
    min_x, max_x = max(min(x_values) - margin, 0), min(max(x_values) + margin, image_array.shape[1])
    min_y, max_y = max(min(y_values) - margin, 0), min(max(y_values) + margin, image_array.shape[0])
    
    # Return the cropped image along with the offsets used for cropping
    return image_array[int(min_y):int(max_y), int(min_x):int(max_x)], min_x, min_y

def overlay_graph_on_image(image_array, G, margin=15):
    """
    Overlays a networkx graph on a BGR image, converting it to grayscale and cropping it.
    
    Parameters:
    - image_array: numpy array of the image in BGR format
    - G: networkx graph with nodes that have 'x' and 'y' attributes for positions
    - margin: margin to use when cropping the image
    
    Returns:
    - The matplotlib figure object with the overlaid graph.
    """
    # Convert the BGR image to grayscale
    grayscale_image = convert_bgr_to_grayscale(image_array)
    
    # Extract positions from the graph node attributes
    positions = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    # Crop the image and get new minimum coordinates
    cropped_image, min_x, min_y = crop_image(grayscale_image, positions, margin)
    
    # Adjust positions for the cropped area
    adjusted_positions = {node: (pos[0] - min_x, pos[1] - min_y) for node, pos in positions.items()}
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(cropped_image.shape[1] / 100, cropped_image.shape[0] / 100), dpi=100)
    # Display the image in grayscale
    ax.imshow(cropped_image, cmap='gray')
    # Draw the graph on the image
    nx.draw(G, adjusted_positions, node_size=20, edge_color='r', node_color='yellow', with_labels=False, ax=ax)
    # Remove the axis
    ax.axis('off')
    
    return fig

# Example usage with a mock image array and graph
# Create a mock image array (e.g., 768x768 pixels)
#make the mock image a random image
mock_image_array = np.random.rand(768, 768, 3)

# Create a graph with 48 nodes and mock coordinates
G_example = nx.Graph()
for i in range(48):
    # Mock coordinates, replace with actual coordinates from your data
    G_example.add_node(i, x=np.random.randint(100, 668), y=np.random.randint(100, 668))

# Add edges between each adjacent node (for visualization purposes)
for i in range(47):
    G_example.add_edge(i, i+1)

# Call the function to overlay the graph on the image
fig = overlay_graph_on_image(mock_image_array, G_example)

# Display the figure
plt.show()  # Note: In an interactive environment, this will display the plot.
# If running as a script, you may need to call `fig.show()` instead.

# Save the figure

output_path = r"C:\Users\User\Desktop\HACKATON\Speech_Recognition_HackCVC_24\src\demo\gray_graph_image.png"

fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close(fig)
