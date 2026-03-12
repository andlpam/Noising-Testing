import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import subprocess
import platform
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
        return data['depth'][1]
      
  def generate_depth_error_image(self, clean_path, noisy_path, noise_type):
    
      clean_depth_frame = self.load_npz_file(clean_path)
      noisy_depth_frame = self.load_npz_file(noisy_path)
      upper_lim_noisy = np.percentile(noisy_depth_frame, 0.95)
      lower_lim_noisy = np.percentile(noisy_depth_frame, 0.05)
      
      cur_noise_frame = np.clip(noisy_depth_frame, lower_lim_noisy, upper_lim_noisy)
      
      mean_noise = np.mean(cur_noise_frame)
      stdev_noise = np.std(cur_noise_frame)
      
      mean_clean = np.mean(clean_depth_frame)
      stdev_clean =np.std(clean_depth_frame)
      
      matrix_noise = (cur_noise_frame - mean_noise)/stdev_noise
      matrix_clean = (clean_depth_frame - mean_clean)/stdev_clean

      error_matrix = np.abs(matrix_clean-matrix_noise)
      
      plt.figure(figsize=(8,6))
      
      depth_map = plt.imshow(error_matrix, cmap='plasma', vmin=0, vmax =1)
      
      plt.colorbar(depth_map, label="Abs Error")
      
      plt.title(f"Error Map: Clean vs {noise_type}")
      
      plt.axis('off')
      
      img_path = f"{self.local_metrics}/error_depthmap_{noise_type}.png"
      
      plt.savefig(img_path, bbox_inches='tight')
      
      plt.close()
      
  def find_right_metrics(self,output):
    
      #Find in the text the right Metrics describing the noise
      mean_val = "Error"
      stdev_val = "Error"
      for line in output.split('\n'):
        
        line_lower = line.lower()
        if "mean distance" in line_lower and "std deviation" in line_lower:
            split_metrics = line.split("/")
            if len(split_metrics) == 2:
              mean_val = split_metrics[0].split('=')[-1].strip()
              stdev_val = split_metrics[1].split('=')[-1].strip()
              
        elif "mean distance" in line_lower:
            split_parts = line.split("=")
            if len(split_parts) > 1:
              mean_val = split_parts[-1].strip()
              
        elif "std deviation" in line_lower:
            split_parts = line.split("=")
            if len(split_parts) > 1:
              stdev_val = split_parts[-1].strip()
      return mean_val, stdev_val
    
  def calculate_3d_metrics(self, clean_dir_path, noise_dir_path, noise_type):

      # Native Linux Paths (for Python)
      clean_glb = glob.glob(os.path.join(clean_dir_path,"*.glb"))[0] 
      noise_glb = glob.glob(os.path.join(noise_dir_path,"*.glb"))[0] 

      linux_clean_ply = clean_glb.replace(".glb", "_extracted.ply")
      linux_noise_ply = noise_glb.replace(".glb", "_extracted.ply")

      print(f"\n -> Asking CloudCompare to analyse noise {noise_type}..")
      
      # Multi-platform path formatter (Linux -> Windows for CC)
      def format_path_for_cc(file_path, cc_exe):
        if platform.system() == "Linux" and cc_exe.lower().endswith(".exe"):
          parts = file_path.split("/")
          if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
            drive = parts[2].upper()
            rest_of_path = "/".join(parts[3:])
            return f"{drive}:/{rest_of_path}"
          
        return os.path.abspath(file_path)
      
      # Helper function to convert GLB to PLY natively using Python!
      def convert_glb_to_ply(linux_glb, linux_ply):
          # Try using Trimesh first
          try:
              import trimesh
              scene = trimesh.load(linux_glb, force='scene')
              
              vertices = []
              for geom in scene.geometry.values():
                  if hasattr(geom, 'vertices'):
                      vertices.extend(geom.vertices)
              
              # If trimesh loaded it directly as a mesh instead of a scene
              if not vertices and hasattr(scene, 'vertices'):
                  vertices = scene.vertices
                  
              pc = trimesh.PointCloud(vertices)
              pc.export(linux_ply)
              return True
              
          except ImportError:
              # Fallback to Open3D if trimesh isn't installed
              try:
                  import open3d as o3d
                  mesh = o3d.io.read_triangle_mesh(linux_glb)
                  if not mesh.has_vertices():
                      pcd = o3d.io.read_point_cloud(linux_glb)
                  else:
                      pcd = o3d.geometry.PointCloud()
                      pcd.points = mesh.vertices
                      
                  o3d.io.write_point_cloud(linux_ply, pcd)
                  return True
                  
              except ImportError:
                  print("\n❌ Error: We need a Python library to bypass CloudCompare's bug.")
                  print("Please run this command in your WSL terminal:")
                  print("pip install trimesh")
                  return False
          except Exception as e:
              print(f"❌ Python conversion failed: {e}")
              return False

      # Execute the native Python conversions
      if not convert_glb_to_ply(clean_glb, linux_clean_ply): return None
      if not convert_glb_to_ply(noise_glb, linux_noise_ply): return None

      # Execute the conversions
      if not convert_glb_to_ply(clean_glb, linux_clean_ply): return None
      if not convert_glb_to_ply(noise_glb, linux_noise_ply): return None

      # Now that we have pure, ungrouped PLYs, format them for Windows CC
      cc_clean_ply_win = format_path_for_cc(linux_clean_ply, self.cc_path)
      cc_noise_ply_win = format_path_for_cc(linux_noise_ply, self.cc_path)

      # The final ICP calculation!
      command = [
        self.cc_path,
        "-SILENT",
        "-AUTO_SAVE", "OFF",
        "-O", cc_clean_ply_win,
        "-O", cc_noise_ply_win,
        "-ICP",
        "-C2C_DIST",
      ]
      
      result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
      out_text = result.stdout
      
      # Parse the output
      mean_val, stdev_val = self.find_right_metrics(out_text)
      
      if mean_val == "Error" or stdev_val == "Error":
        print(f"❌ Metrics parsing failed for {noise_type}..")
        print("--- CC ICP Output ---")
        print(out_text)
        print("---------------------")
        return None
      
      return {
        "Noise" : noise_type,
        "Mean Distance" : mean_val,
        "Std Deviation" : stdev_val,
      }
      
    
    
      
    
      
  
    
  
      
      
      
  
