import os
import torch
from depth_anything_3.api import DepthAnything3
from noise import Noise
from pathlib import Path
from helpers import create_clean_dirs
import json
import argparse
import glob
import numpy as np
#Data directory... Different for every person
FRAME_CHOOSEN = 4
images_extensions = ['*.png', '*.jpg']

class DA3Runner:
    def __init__(self, input_path, output_dir ,metrics_dir,fps, noise_types, seed=5436364):
        self.input_path = input_path
        self.output_dir = output_dir
        self.fps = fps
        self.noise_types = noise_types
        self.seed = seed
        self.metrics_dir = metrics_dir
        
        print("-> Loading DA3 Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to(self.device)
        
    def find_dirs_os(self):
        
        dirs_with_images = []
        #Collect directories with image
        for root, _, files in os.walk(self.input_path):
            #Verify if it exists images efficiently
            if any(f.lower().endswith(tuple(ext.replace('*','') for ext in images_extensions)) for f in files):
                dirs_with_images.append(root)
                    
        return dirs_with_images

    def prepare_images(self, dirs_with_images):
        dir_images = {}
        valid_exts = [ext.replace('*', '') for ext in images_extensions]
        for dir in dirs_with_images:
            cur_dir = Path(dir)
            img_list = [
                str(f) for f in cur_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
            
            if len(img_list) == 0:
                print(f"No images found in {dir}")
            
            dir_images[f"{os.path.basename(dir)}"] = img_list
            
        return dir_images
      
# Main Function
    def run_inference(self):
        #Preparing videos to be processed by DA3
        noise = Noise(self.seed, self.noise_types, self.input_path, self.fps)
        noise.apply_noise()
        
        dirs_path_with_images = self.find_dirs_os()
        
        dir_images = self.prepare_images(dirs_path_with_images)
        
        print("Loading Da3 model..")
    
        
        noise_documentation = {}
        #Clean reconstructions before
        create_clean_dirs(self.output_dir)
        for dir_name, img_list in dir_images.items():
            noise_full_details = {}
            full_path_out = os.path.join(self.output_dir, dir_name)
            
            prediction = self.model.inference(
                image = img_list,
                ref_view_strategy = "middle",
                process_res = 140,
                use_ray_pose=False,
                process_res_method = "upper_bound_resize",
                export_dir = full_path_out,
                export_format = "glb",
                show_cameras=False, #Cloud Compare cannot do ICP if this is turned on
            )
            
            npz_path = os.path.join(full_path_out, "depth_data.npz")
            
            np.savez_compressed(
                npz_path,
                depth=prediction.depth
            )
            
            #DEBUG-----------------------
            npz_file = glob.glob(os.path.join(full_path_out,"*.npz"))[0]
            glb_file = glob.glob(os.path.join(full_path_out,"*.glb"))[0]
             
            if npz_file and glb_file:
                print("FILES SAVED WITH SUCCESS!!")
            else:
                print("Some files were not saved with success!!")
                
            #The input args might not match 
            target_type = next((type for type in self.noise_types if type in dir_name), None)
            #Put the details in the dictionary--------------------------
            
            noise_full_details["input_dir_path"] = full_path_out
            noise_full_details["Normal Depth Map"] = glob.glob(os.path.join(full_path_out,"depth_vis", f"*{FRAME_CHOOSEN}*.jpg"))[0]
            noise_full_details["Depth Map Error"] = os.path.join(self.metrics_dir, f"error_depthmap_{target_type}.png")
            noise_full_details["3D Reconstruction"] = os.path.join(self.metrics_dir, f"reconstruction_{target_type}.jpg") #Not yet created
            
            noise_documentation[target_type] = noise_full_details
            
                
            
        #Clean garbage
        torch.cuda.empty_cache()
        
        return noise_documentation
        
#Running DA3Runner and saving the object in json in the metrics folder.
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/app/data/videos_in')
    parser.add_argument('--output', type=str, default='/app/data/reconstructions_out')
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--noises', nargs='+', default=['clean', 'awgn', 'salt_and_pepper', 'shot_noise', 'speckle_noise'])
    parser.add_argument('--metrics',type=str,default='/app/data/metrics_results')
    args = parser.parse_args()
    
    print("-> [DOCKER] Initializing 3D extraction...")
    runner = DA3Runner(args.input, args.output, args.metrics,args.fps, args.noises)
    out_dirs_dict = runner.run_inference()
    json_dict_path = os.path.join(args.metrics, "inference_output.json")
    
    with open(json_dict_path, "w") as f:
        json.dump(out_dirs_dict,f)
    
    
    