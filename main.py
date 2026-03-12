import argparse
import os
from helpers import create_clean_dirs, turn_relative_path_into_full
from evaluate_results import MetricsEval
import csv
import subprocess
import json
if __name__ == "__main__":
    
    #GETTING THE INPUT VALUES REQUIRED ------------------------------------
    parser = argparse.ArgumentParser(description="DA3 Reconstruction and Noising Pipeline")
    CHOISES_FOR_VIDEO_TYPES = ['clean', 'awgn', 'salt_and_pepper', 'shot_noise', 'speckle_noise']
    parser.add_argument(
        '--local_data',
        type=str,
        required=True,
        help='Path to the local data (parent of videos_in)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='Fps for doing the reconstruction of the images'
    )
    parser.add_argument(
        '--noises', 
        type=str,
        nargs='+', # Accept one or more arguments
        choices=CHOISES_FOR_VIDEO_TYPES,
        default=CHOISES_FOR_VIDEO_TYPES, # O default caso o professor não passe nada
        help='Types of video/noises to inject in reconstructions (ex: --types clean awgn shot_noise)'
    )
    parser.add_argument(
       '--cc_path',
       type=str,
       default=r"C:\Program Files\CloudCompare\CloudCompare.exe",
       help="Cloud compare path to do metric analysis."
    )
    args = parser.parse_args()
    
    local_output = os.path.join(args.local_data, "reconstructions_out")
    local_metrics = os.path.join(args.local_data, "metrics_results")
    
    create_clean_dirs(local_output)
    create_clean_dirs(local_metrics)
    
    abs_local_data = os.path.abspath(args.local_data)
    # RUNNING GLB AND NPZ FILES IN DOCKER---------------------------
    docker_command = [
        "docker", "run", "--gpus", "all", "-it", "--rm",
        "--mount", f"type=bind,source={abs_local_data},target=/app/data",
        "da3-gpu",
        "--fps", str(args.fps),
        "--noises"
    ] + args.noises # add noises
    #RUN DOCKER FILE
    subprocess.run(docker_command)
    
    #Read PATH DICTIONARY OBJECT----------------------------------
    
    json_path = os.path.join(local_metrics, "inference_output.json")
    
    if not os.path.exists(json_path):
        print(f"json file doesnt exist..")
        exit(1)
    
    with open(json_path, "r") as f:
        paths_dict = json.load(f)
        
    #GENERATING METRICS-------------------------------------------
    
    metrics_evaluator = MetricsEval(args.cc_path, local_metrics)
    full_path_clean = turn_relative_path_into_full(paths_dict["clean"], local_output)
    every_result = []
    for dir_name, dir_path in paths_dict.items():
        
        if dir_name == "clean":
            continue
        
        full_path_noise = turn_relative_path_into_full(dir_path, local_output)
        
        metrics_evaluator.generate_plasma_image(full_path_clean, full_path_noise, dir_name)
        
        noise_result = metrics_evaluator.calculate_3d_metrics(full_path_clean, full_path_noise, dir_name)
        
        if noise_result:
            every_result.append(noise_result)
            print(f"Metrics of {dir_name} saved..")
    
    #SAVING METRICS IN A CSV--------------------------------------
    csv_path = os.path.join(local_metrics, "metrics_results.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        # Define columns in excel
        columns = ["Noise", "Mean Distance", "Std Deviation"]
        
        # Create a writer in excel
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        
        #Write excel header
        writer.writeheader()
    
        writer.writerows(every_result)
        
    print(f"\n Result table created with success in {csv_path}!")
            
    
   