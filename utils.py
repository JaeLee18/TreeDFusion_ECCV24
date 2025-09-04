import cv2
import numpy as np

def compute_sharpness_values(image, patch_size=32, stride=16):
    """
    Compute sharpness values for an image using patches.
    
    Parameters:
    - image: PIL Image object (grayscale)
    - patch_size: Size of the square patch
    - stride: Stride for the sliding window
    
    Returns:
    - List of sharpness values for each patch
    """
    sharpness_values = []
    image_np = np.array(image)
    width, height = image.size

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image_np[y:y+patch_size, x:x+patch_size]
            laplacian = cv2.Laplacian(patch, cv2.CV_64F)
            sharpness = np.var(laplacian)
            sharpness_values.append(sharpness)

    return sharpness_values
 