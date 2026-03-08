import cv2
import os
import glob
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from noise import Noise
from pathlib import Path
#Data directory... Different for every person
INPUT_PATH = '/mnt/e/denoising_and_reconstruction/videos_in'
OUTPUT_DIR = 'reconstructions_out'
METRICS_DIR = 'metrics_results'
SEED_NUMBER = 5436364
images_extensions = ['*.png', '*.jpg']
types_of_videos = ['clean','awgn', 'salt_and_pepper', 'shot_noise', 'speckle_noise']
parent_dir = os.path.dirname(os.path.normpath(INPUT_PATH))
output_path = os.path.join(parent_dir, OUTPUT_DIR)
metrics_path = os.path.join(parent_dir, METRICS_DIR)

def find_dirs_os(parent_dir):
    
    dirs_with_images = []
    #Collect directories with image
    for root, _, files in os.walk(parent_dir):
        #Verify if it exists images efficiently
        if any(f.lower().endswith(images_extensions) for f in files):
            dirs_with_images.append(root)
                
    return dirs_with_images

def prepare_images(dirs_with_images):
    dir_images = {}
    for dir in dirs_with_images:
        cur_dir = Path(dir)
        img_list = [
            str(f) for f in cur_dir.iterdir()
            if f.is_file() and f.suffix.lower() in images_extensions
        ]
        
        if len(img_list) == 0:
            print("No images found")
        
        dir_images[f"{os.path.basename(dir)}"] = img_list
        
    return dir_images
        
        

# Main Function
def run_inference():
    #Preparing videos to be processed by DA3
    noise = Noise(SEED_NUMBER, types_of_videos, INPUT_PATH)
    noise.apply_noise()
    
    dirs_path_with_images = find_dirs_os(INPUT_PATH)
    
    dir_images = prepare_images(dirs_path_with_images)
        
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
    model = model.to(device=device)
    
    for dir_name, img_list in dir_images.items():
        _ = model.inference(
            image = img_list,
            ref_view_strategy = "middle",
            process_res = 336,
            use_ray_pose=False,
            process_res_method = "upper_bound_resize",
            export_dir = os.path.join(output_path, dir_name),
            export_format = "glb",
        )
        
    