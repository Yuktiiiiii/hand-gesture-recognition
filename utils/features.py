# Convex hull, finger detection, and gesture recognition.
# This module provides functions to detect convex hulls, identify fingers, and recognize gestures in images.
# utils/features.py
import cv2
import numpy as np

def extract_features(image):
    """
    Extracts features from a preprocessed binary hand image.

    Features:
    - Number of fingers (using convexity defects)
    - Hu moments (7 shape descriptors)

    Parameters:
        image (np.array): Preprocessed binary image (white hand on black background).

    Returns:
        features (list): [finger_count, hu1, hu2, ..., hu7]
    """
    features = []

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0] + [0]*7  # No hand detected

    # Assume the largest contour is the hand
    contour = max(contours, key=cv2.contourArea)

    # Convex hull and defects for finger counting
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is not None and len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Calculate angle using cosine rule
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-5))  # +epsilon to avoid divide-by-zero

                if angle <= np.pi / 2:  # less than 90 degrees
                    finger_count += 1
        features.append(finger_count)

    else:
        features.append(0)

    # Hu Moments
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Convert to log scale for better numerical stability
    hu_moments = [-np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu_moments]
    features.extend(hu_moments)

    return features