#This file will take an image of lips and a graph and will overlap the graph on the image
# Function to overlay a networkx graph on an image

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def overlay_graph_on_image(image_array, G):
    """
    Overlays a networkx graph on an image.
    
    Parameters:
    - image_array: numpy array of the image
    - G: networkx graph with nodes that have 'x' and 'y' attributes for positions
    
    Returns:
    - The matplotlib figure object with the overlaid graph.
    """
    # Extract positions from the graph node attributes
    positions = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(image_array.shape[1] / 100, image_array.shape[0] / 100), dpi=100)
    # Display the image
    ax.imshow(image_array)
    # Draw the graph on the image
    nx.draw(G, positions, node_size=20, edge_color='r', node_color='yellow', with_labels=False, ax=ax)
    # Remove the axis
    ax.axis('off')
    
    return fig
if __name__ == "__main__":
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
    output_path = "/home/GROUP02/Speech_Recognition_HackCVC_24/src/demo/overlaid_graph_image.png"
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


