import argparse
import os
from helpers import create_clean_dirs
from evaluate_results import MetricsEval

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
        help='Path to the local input data (videos or images)'
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
    
    local_dir = os.path.dirname(args.local_data)
    local_output = os.path.join(local_dir, "reconstructions_out")
    local_metrics = os.path.join(local_dir, "metrics_results")
    create_clean_dirs(local_output)
    create_clean_dirs(local_metrics)
    
    
    abs_local_data = os.path.abspath(local_dir)
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
    print(local_dir)
    print(local_output)
    print(local_metrics)
    json_path = os.path.join(local_metrics, "inference_output.json")
    
    if not os.path.exists(json_path):
        print(f"json file doesnt exist..")
        exit(1)
    
    with open(json_path, "r") as f:
        noise_documentation = json.load(f)
        
    #GENERATING METRICS-------------------------------------------
    
    metrics_evaluator = MetricsEval(args.cc_path, local_metrics, local_output, args.local_data)
    metrics_evaluator.run_evaluation_pipeline(noise_documentation)
            
    
   