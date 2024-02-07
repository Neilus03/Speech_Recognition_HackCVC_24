'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def merge_views(face_with_boundingbox, lips_with_graph1, lips_with_graph2, transcription):
    """
    Merges the face image, the lip graph, and the transcription into a single horizontal image.
    Uses cv2, creates a black background and pastes the images and text on it.
    The height is the max height of the two images plus some padding for the text.
    The width is the sum of face_with_boundingbox width and lips_with_graph width.
    
    Parameters:
    - face_with_boundingbox: numpy array of the face image with the bounding box drawn
    - lips_with_graph: numpy array of the lips image with the graph overlaid
    - transcription: string with the transcription
    
    Returns:
    - The merged image as a numpy array.
    """
    
    # Determine dimensions for the merged image
    total_height = max(face_with_boundingbox.shape[0], lips_with_graph1.shape[0]) + 50  # 50 pixels for padding for text
    total_width = face_with_boundingbox.shape[1] + lips_with_graph1.shape[1]
    
    # Create a black background image
    merged_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Paste the face image on the left
    merged_image[:face_with_boundingbox.shape[0], :face_with_boundingbox.shape[1]] = face_with_boundingbox
    
    # Paste the lips image with the graph to the right of the face image
    merged_image[:lips_with_graph1.shape[0], face_with_boundingbox.shape[1]:face_with_boundingbox.shape[1] + lips_with_graph1.shape[1]] = lips_with_graph1
    
    # Paste the second lips image with the graph to the right of the face image
    merged_image[:lips_with_graph2.shape[0], face_with_boundingbox.shape[1] + lips_with_graph1.shape[1]:face_with_boundingbox.shape[1] + lips_with_graph1.shape[1] + lips_with_graph2.shape[1]] = lips_with_graph2
    
    # Add the transcription as text on the bottom across the width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    
    # Calculate text position to be at the bottom center of the merged image
    text_size = cv2.getTextSize(transcription, font, font_scale, font_thickness)[0]
    text_x = (total_width - text_size[0]) // 2  # center the text
    text_y = total_height - 10  # 10 pixels from the bottom

    cv2.putText(merged_image, transcription, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    return merged_image


if __name__ == "__main__":
    # Create mock images using random noise for demonstration
    mock_face_with_boundingbox = np.random.randint(255, size=(500, 500, 3), dtype=np.uint8)
    mock_lips_with_graph1 = np.random.randint(255, size=(200, 400, 3), dtype=np.uint8)
    mock_lips_with_graph2 = np.random.randint(255, size=(200, 400, 3), dtype=np.uint8)
    mock_transcription = "hello world!"

    # Call the merge_views function
    merged_image = merge_views(mock_face_with_boundingbox, mock_lips_with_graph1, mock_lips_with_graph2, mock_transcription)

    # Display the merged image using matplotlib
    plt.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))  # Convert color from BGR to RGB
    plt.axis('off')  # Hide axes

    plt.savefig('merged_views.png', bbox_inches='tight', pad_inches=0)  # Save the image
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

def merge_views(face_with_boundingbox, lips_with_graph1, lips_with_graph2, transcription):
    """
    Merges the face image, two lip graphs, and the transcription into a single horizontal image.
    Uses cv2, creates a black background and pastes the images and text on it.
    The height is the sum of the max height of the face image and lips_with_graph2 plus some padding for the text.
    The width is the sum of face_with_boundingbox width and the max width of the two lips graphs.
    
    Parameters:
    - face_with_boundingbox: numpy array of the face image with the bounding box drawn
    - lips_with_graph1: numpy array of the first lips image with the graph overlaid
    - lips_with_graph2: numpy array of the second lips image with the graph overlaid
    - transcription: string with the transcription
    
    Returns:
    - The merged image as a numpy array.
    """
    
    # Determine dimensions for the merged image
    lips_max_height = lips_with_graph1.shape[0] + lips_with_graph2.shape[0]
    total_height = max(face_with_boundingbox.shape[0], lips_max_height) + 50  # 50 pixels for padding for text
    total_width = face_with_boundingbox.shape[1] + max(lips_with_graph1.shape[1], lips_with_graph2.shape[1])
    
    # Create a black background image
    merged_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Paste the face image on the left
    merged_image[:face_with_boundingbox.shape[0], :face_with_boundingbox.shape[1]] = face_with_boundingbox
    
    # Paste the second lips image with the graph above the first one
    merged_image[:lips_with_graph2.shape[0], face_with_boundingbox.shape[1]:face_with_boundingbox.shape[1] + lips_with_graph2.shape[1]] = lips_with_graph2
    
    # Paste the first lips image with the graph below the second one
    merged_image[lips_with_graph2.shape[0]:lips_max_height, face_with_boundingbox.shape[1]:face_with_boundingbox.shape[1] + lips_with_graph1.shape[1]] = lips_with_graph1
    
    # Add the transcription as text on the bottom across the width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    
    # Calculate text position to be at the bottom center of the merged image
    text_size = cv2.getTextSize(transcription, font, font_scale, font_thickness)[0]
    text_x = (total_width - text_size[0]) // 2  # center the text
    text_y = total_height - 10  # 10 pixels from the bottom

    cv2.putText(merged_image, transcription, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    return merged_image


if __name__ == "__main__":
    # Create mock images using solid colors for demonstration
    mock_face_with_boundingbox = np.full((500, 500, 3), (255, 0, 0), dtype=np.uint8)  # Red color
    mock_lips_with_graph1 = np.full((200, 400, 3), (0, 255, 0), dtype=np.uint8)  # Green color
    mock_lips_with_graph2 = np.full((200, 400, 3), (0, 0, 255), dtype=np.uint8)  # Blue color
    mock_transcription = "hello world!"

    # Call the merge_views function
    merged_image = merge_views(mock_face_with_boundingbox, mock_lips_with_graph1, mock_lips_with_graph2, mock_transcription)

    # Display the merged image using matplotlib
    plt.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))  # Convert color from BGR to RGB
    plt.axis('off')  # Hide axes

    plt.savefig('merged_views.png', bbox_inches='tight', pad_inches=0)  # Save the image

