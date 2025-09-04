from glob import glob
from tqdm import tqdm
import os
import torch
import argparse
from datetime import datetime
import time
import logging
from PIL import Image
import shutil
import numpy as np
from utils import compute_sharpness_values
import sys

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Tree-D Fusion')

    # Add an argument for the list of items to be saved
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu_id, and the default is the first gpu, which is 0.')

    parser.add_argument('--image_path', type=str,
                        help='A path to the image file.')

    parser.add_argument('--tree_species', type=str,default="",
                        help='Tree species name.')
    parser.add_argument('--city', type=str, default='myCity',
                        help='A city name.')

    parser.add_argument('--start_idx', type=int,
                        help='data image start idx')
    parser.add_argument('--end_idx', type=int,
                        help='data image end idx')
    args = parser.parse_args()

    image_path = args.image_path
    city = args.city
    save_path = f"{city}_outputs"
    gpu_id = args.gpu_id
    tree_species = args.tree_species

    cnt = 0
    start = args.start_idx
    end = args.end_idx
    # Create and configure logger
    logging.basicConfig(filename=f"gpu{gpu_id}_imagepath{start}_end{end}.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)    

    print(f"Tree-D Fusion started with {image_path}")
    if tree_species == "":
        prompt = "_tree_"
    else:
        prompt = f"_{tree_species}_tree_"
    logger.info(f"{image_path} started")

    # Step1. Segment Anything
    print("Step1. Starting Semgenting a tree from the image.")
    start = time.time()
    basename = os.path.basename(image_path).split(".")[0]
    save_path_base = f"{save_path}/{basename}"
    os.makedirs(save_path, exist_ok=True)
    os.system(f'python segment.py --img_path {image_path} --save_path {save_path_base}')
    segmented_image = f"{save_path_base}/{basename}_segmented.png"
    seg_end = time.time()-start
    logger.info(f"Segmentation ended|{seg_end} seconds")
    


    # Step2. Preprocess Image
    print("Step2. Extracting tree mask from the segmented image.")
    os.system(f'python extract-mask.py --image_path {segmented_image} --output_dir {save_path_base}')
    clean_image = f"{save_path_base}/rgba.png"
    mask_end = time.time()-start
    logger.info(f"mask ended|{mask_end} seconds")
    
    image_new = Image.open(clean_image)
    # Convert to grayscale
    gray_image_new = image_new.convert('L')
    vals = compute_sharpness_values(gray_image_new)
    if(np.mean(vals) < 5000):
        logger.info(f"{clean_image} blurry, {np.mean(vals)}")
        print("Image is too blurry, lower the sharpness threshold (5000) or try with another image.")
        sys.exit(1)
    

    # Step3. Generate Tree
    RUN_ID = basename
    torch.cuda.empty_cache()

    now = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    logger.info(f"clean_image: {clean_image}")
    print("Step3. Now, we are running the AI model to generate a 3D model.")
    
    cmd = f'python Magic123/main.py -O\
    --text {prompt}\
    --sd_version 1.5\
    --image {clean_image}\
    --workspace {save_path}/{RUN_ID}\
    --optim adam\
    --iters 2000\
    --h 64\
    --w 64\
    --mcubes_resolution 128\
    --guidance SD zero123\
    --lambda_guidance 1.0 8.0\
    --guidance_scale 20.0 5.0\
    --latent_iter_ratio 0\
    --t_range 0.2 0.6\
    --lambda_depth 0\
    --lambda_depth_mse 0\
    --lambda_normal 0\
    --lambda_normal_smooth 0\
    --lambda_normal_smooth2d 0\
    --lambda_mask 20.0\
    --lambda_rgb 5.0\
    --lambda_opacity 0.01\
    --bg_radius -1\
    --blob_density 8 --blob_radius 0.35\
    --save_mesh\
    --zero123_config zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml\
    --zero123_ckpt ./tree.ckpt'

    os.system(cmd)
    gen_end = time.time()-start
    logger.info(f"mesh training ended|{gen_end} seconds")
    
    image_path = f"{save_path}/{RUN_ID}/rgba.png"
    mesh_path = f"{save_path}/{RUN_ID}/mesh/mesh.obj"
    image_object_name = f"{city}/{tree_species}_{RUN_ID}/rgba.png"
    obj_object_name = f"{city}/{tree_species}_{RUN_ID}/mesh.obj"
    print("\n" + "="*40)
    print(f"  📁 Saved Paths & Object Names 📁")
    print("="*40)
    print(f"  🖼  Image Path       : {image_path}")
    print(f"  🗿  Mesh Path        : {mesh_path}")
    print(f"  🎨  Image Object Name: {image_object_name}")
    print(f"  🏗  OBJ Object Name  : {obj_object_name}")
    print("="*40 + "\n")

