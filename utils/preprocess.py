import cv2
import numpy as np

def preprocess_image(image, threshold_value=150):
    """
    Simple preprocessing for B/W hand gesture images.
    Converts all gray pixels above threshold to white, others to black.

    Parameters:
        image (np.array): Grayscale input image.
        threshold_value (int): Pixel value above which it becomes white (255).

    Returns:
        binary (np.array): Thresholded binary image.
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply simple binary thresholding
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    return thresholded_image
