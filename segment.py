import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os

import argparse

def extract_mask_region_opencv_image(original_image_np, mask_tensor, save_path):
    # Convert the PyTorch tensor mask to a numpy array
    mask_np = (mask_tensor > 0.5).astype(np.uint8)  # Convert to binary mask with values 0 or 1

    # If the mask is a single channel (grayscale), we repeat it to have 3 channels for compatibility with the original image
    mask_np = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)

    # Use the mask to extract the region of interest from the OpenCV image
    result_np = original_image_np * mask_np

    # Convert numpy array to PIL Image and save
    result_img = Image.fromarray(result_np)
    result_img.save(save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save arguments to a text file.')

    # Add an argument for the list of items to be saved
    parser.add_argument('--img_path', type=str,
                        help='an image path to segment')
    parser.add_argument('--save_path', type=str,
                        help='a path to save the segmented image')

    args = parser.parse_args()
    
    image_path = args.img_path
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    if(os.path.exists(sam_checkpoint) == False):
        print(f"[segment.py] Segment Anything model not found: {sam_checkpoint}")
        print(f"[segment.py] Image segmentation failed")

    model_type = "vit_h"
    device = "cuda:0"
    save_folder = args.save_path
    os.makedirs(save_folder, exist_ok=True)


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    (h, w) = image.shape[:2]
    input_point = np.array([[w//2, h//2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # Assuming you have the image loaded via OpenCV as opencv_image and the tensor mask as mask_tensor:
    image_name = os.path.basename(image_path).split(".")[0]
    extract_mask_region_opencv_image(image, masks[0], f'{save_folder}/{image_name}_segmented.png')
