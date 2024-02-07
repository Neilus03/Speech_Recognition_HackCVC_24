# This  function receives an image as a np array and 4 coordinates of a bounding box and returns the image with the bounding box drawn on it.

import cv2

def face_boundingbox(image, x, y, w, h):
    """
    Draws a bounding box on an image.
    
    Parameters:
    - image: numpy array of the image
    - x, y: Coordinates of the top-left corner of the bounding box
    - w, h: Width and height of the bounding box
    
    Returns:
    - The image with the bounding box drawn.
    """
    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0.8, 0., 0.), 2)
    
    # Now image is a numpy array with the bounding box drawn
    return image
    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # Example usage of the face_boundingbox function
    # Create a mock image array (e.g., 768x768 pixels)
    mock_image_array = np.random.rand(768, 768, 3)
    
    # Create mock coordinates for a bounding box
    x, y, w, h = 100, 100, 200, 300
    
    # Call the function to draw the bounding box on the image
    image_with_bounding_box = face_boundingbox(mock_image_array, x, y, w, h)
    
    # Display the image with the bounding box
    plt.imshow(image_with_bounding_box)
    plt.imsave('face_bounding_box.png', image_with_bounding_box)
    plt.axis('off')
    plt.show()