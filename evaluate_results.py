import numpy as np
import matplotlib as plt
import os
import glob
import subprocess
class MetricsEval:
  
  def __init__(self, cc_path, local_metrics):
     self.cc_path = cc_path
     self.local_metrics = local_metrics

  def load_npz_file(self,dir_path):
    # Find npz files
    search = os.path.join(dir_path, "*.npz")
    file_list = glob.glob(search)
    
    if not file_list:
        print(f"Error: Didn't found any NPZ file {file_list}")
        return None
    
    final_path = file_list[0]
    
    with np.load(final_path) as data:
        # Return the dimension of a single frame depth
        return data['depth'][0]
      
  def generate_plasma_image(self, clean_dir_path, noise_dir_path, noise_type):
      
      clean_frame = self.load_npz_file(clean_dir_path)
      
      noise_frame = self.load_npz_file(noise_dir_path)
      vmax = max(noise_frame.max(), clean_frame.max())
      vmin = min(noise_frame.min(), clean_frame.min())
    
      _, axes = plt.subplots(1,2)
  
      axes[0].imshow(clean_frame, cmap='plasma', vmin=vmin, vmax=vmax)
      axes[0].set_title("Clean video")
      
      axes[1].imshow(noise_frame, cmap='plasma', vmin=vmin, vmax=vmax)
      axes[1].set_title(f"Noise: {noise_type}")
      
      plt.savefig(f"{self.local_metrics}/comparation_{noise_type}.png")
      
      plt.close()
      
  def find_right_metrics(self,output):
    
      #Find in the text the right Metrics describing the noise
      mean_val = "Error"
      stdev_val = "Error"
      for line in output.split('\n'):
        if "Mean distance" in line:
            split_parts = line.split("=")
            if len(split_parts) > 1:
              mean_val = split_parts[-1].strip()
              
        elif "Std deviation" in line:
            split_parts = line.split("=")
            if len(split_parts) > 1:
              stdev_val = split_parts[-1].strip()
      return mean_val, stdev_val
    
  def calculate_3d_metrics(self, clean_dir_path, noise_dir_path, noise_type):

      #There is only one glb file 
      clean_glb = glob.glob(os.path.join(clean_dir_path,"*.glb"))[0] 
      noise_glb = glob.glob(os.path.join(noise_dir_path,"*.glb"))[0] 

      print(f"\n -> Asking CloudCompare to analyse noise {noise_type}..")
      
      commmand = [
        self.cc_path,
        "-SILENT",
        "-O", clean_glb,
        "-O", noise_glb,
        "-ICP",
        "-C2C_DIST",
        "-POP_CLOUDS"
      ]
      
      result = subprocess.run(commmand, capture_output=True, text=True)
      
      out_text = result.stdout
      
      #Find in the text the right Metrics describing the noise
      mean_val, stdev_val = self.find_right_metrics(out_text)
      
      if mean_val == "Error" or stdev_val == "Error":
        print("Metrics gave an error..")
        return
      
      return {
        "Noise" : noise_type,
        "Mean Distance" : mean_val,
        "Std Deviation" : stdev_val,
      }
      
    
      
      
    
      
  
    
  
      
      
      
  
