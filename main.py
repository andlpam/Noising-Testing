import argparse
import os
from helpers import create_clean_dirs
from run_reconstructions import DA3Runner
from evaluate_results import MetricsEval
import csv
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DA3 Reconstruction and Noising Pipeline")
    CHOISES_FOR_VIDEO_TYPES = ['clean', 'awgn', 'salt_and_pepper', 'shot_noise', 'speckle_noise']
    parser.add_argument(
        '--input', 
        type=str, 
        default='/app/data/videos_in', 
        help='Path to the videos that are going to be processed.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/app/data/metrics_results',
        help='Path to save the output reconstructions and images'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='/app/data/metrics_results',
        help='Path to save the metrics'
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
    
    create_clean_dirs(args.output)
    create_clean_dirs(args.metrics)
    
    runner = DA3Runner(args.input, args.output, args.fps, args.noises)
    out_dirs_dict = runner.run_inference()
    metrics_evaluator = MetricsEval(args.cc_path)
    #Save every result to put on a csv
    
    every_result = []
    for dir_name, dir_path in out_dirs_dict.items():
        
        if dir_name == "clean":
            continue
        
        metrics_evaluator.generate_plasma_image(out_dirs_dict["clean"], dir_path, dir_name)
        
        noise_result = metrics_evaluator.calculate_3d_metrics(out_dirs_dict["clean"], dir_path, dir_name)
        
        if noise_result:
            every_result.append(noise_result)
            print(f"Metrics of {dir_name} saved..")
    
    csv_path = os.path.join(args.metrics, "metrics_results.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        # Define columns in excel
        columns = ["Noise", "Mean Distance", "Std Deviation"]
        
        # Create a writer in excel
        escritor = csv.DictWriter(csv_file, fieldnames=columns)
        
        #Write excel header
        escritor.writeheader()
    
        escritor.writerows(every_result)
        
    print(f"\n Result table created with success in {csv_path}!")
            
    
   