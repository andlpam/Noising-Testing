import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import subprocess
import platform
import open3d as o3d
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helpers import turn_relative_path_into_full
import csv
FRAME_CHOOSEN = 4

class MetricsEval:
  
  def __init__(self, cc_path, local_metrics, local_output, local_input):
     self.cc_path = cc_path
     self.local_metrics = local_metrics
     self.local_output = local_output
     self.local_input = local_input

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
        return data['depth'][FRAME_CHOOSEN]
    
  def load_image(self,path):
    # Se os caminhos não existirem, criamos uma imagem vazia (dummy) para o código não falhar
    try:
        img = Image.open(path).convert('RGB')
        return np.array(img)
    except FileNotFoundError:
        # Retorna uma imagem preta de placeholder se o ficheiro não existir
        return np.zeros((200, 200, 3), dtype=np.uint8)
      
  def generate_depth_error_image(self, clean_path, noisy_path, noise_type):
    
      clean_depth_frame = self.load_npz_file(clean_path)
      noisy_depth_frame = self.load_npz_file(noisy_path)
      vmax = np.percentile(noisy_depth_frame, 95)
      lower_lim_noisy = np.percentile(noisy_depth_frame, 5)
      
      # cur_noise_frame = np.clip(noisy_depth_frame, lower_lim_noisy, upper_lim_noisy)
      
      mean_noise = np.mean(noisy_depth_frame)
      stdev_noise = np.std(noisy_depth_frame)
      
      mean_clean = np.mean(clean_depth_frame)
      stdev_clean = np.std(clean_depth_frame)
      
      matrix_noise = (noisy_depth_frame - mean_noise)/stdev_noise
      matrix_clean = (clean_depth_frame - mean_clean)/stdev_clean

      error_matrix = np.abs(matrix_clean-matrix_noise)
      
      plt.figure(figsize=(8,6))
      
      depth_map = plt.imshow(error_matrix, cmap='plasma', vmin=0,vmax= vmax)
      
      plt.colorbar(depth_map, label="Abs Error")
      
      plt.title(f"Error Map: Clean vs {noise_type}")
      
      plt.axis('off')
      
      img_path = os.path.join(self.local_metrics, f"error_depthmap_{noise_type}.png")#f"{self.local_metrics}/error_depthmap_{noise_type}.png"
      
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
  """ Using the cpu to process the screenshot due to wsl"""
  def generate_3d_screenshot(self, glb_path, output_img_path):
      print(f"📸 A gerar screenshot para: {os.path.basename(glb_path)}")
      
      # 1. FORÇAR O WSL A IGNORAR A GPU PARA RENDERING E USAR SOFTWARE (Evita erros ZINK/Wayland)
      os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
      os.environ["GALLIUM_DRIVER"] = "llvmpipe"
      
      try:
          import trimesh
          import open3d.visualization.rendering as rendering
          
          # 2. Carregar a nuvem de pontos com Trimesh (Robusto para os GLBs do DA3)
          scene = trimesh.load(glb_path, force='scene')
          
          vertices = []
          colors = []
          for geom in scene.geometry.values():
              if hasattr(geom, 'vertices'):
                  vertices.extend(geom.vertices)
              if hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors'):
                  colors.extend(geom.visual.vertex_colors[:, :3])
          
          if not vertices and hasattr(scene, 'vertices'):
              vertices = scene.vertices
              if hasattr(scene, 'visual') and hasattr(scene.visual, 'vertex_colors'):
                  colors = scene.visual.vertex_colors[:, :3]

          if len(vertices) == 0:
              print("❌ O GLB não tem pontos válidos.")
              return None

          # 3. Converter para nuvem de pontos Open3D
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(np.array(vertices))
          
          if len(colors) == len(vertices):
              pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)
          else:
              pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Cinza se não houver cores

          # 4. Usar o OffscreenRenderer do Open3D (Feito para servidores e WSL sem interface gráfica)
          # Resolução: 1024x1024
          render = rendering.OffscreenRenderer(1024, 1024)
          
          # Material básico para os pontos brilharem e não ficarem pretos
          mat = rendering.MaterialRecord()
          mat.shader = "defaultUnlit"
          mat.point_size = 2.0 # Tamanho do ponto (ajusta se ficar muito denso/fino)
          
          render.scene.add_geometry("pcd", pcd, mat)
          render.scene.set_background([1.0, 1.0, 1.0, 1.0]) # Fundo Branco (RGBA)
          
          # 5. Posicionar a câmara automaticamente para ver todos os pontos
          bounds = pcd.get_axis_aligned_bounding_box()
          center = bounds.get_center()
          # Ajusta a câmara: (campo de visão, ponto de foco, posição da câmara, vetor "Cima")
          # Pode ser necessário afinar a posição [center[0], center[1], center[2] - 3] dependendo dos eixos do DA3
          render.setup_camera(60.0, center, [center[0], center[1], center[2] - 3], [0, -1, 0])
          
          # 6. Renderizar imagem silenciosamente e guardar
          img = render.render_to_image()
          o3d.io.write_image(output_img_path, img)
          
          print(f"✅ Screenshot guardado com sucesso em {output_img_path}")
          
      except Exception as e:
          print(f"❌ Erro ao gerar screenshot 3D: {e}")
          
      return output_img_path
  
  """
  This method plots in an html page the results of the injecting noise
  
  Pre conditions:
  -All Images must be Ploted in their corresponding file paths
  """
  def plot_page(self, dict_of_noise):
      #Remove clean here because we dont need it
      _ = dict_of_noise.pop("clean", None)
      num_rows = len(dict_of_noise.keys())
      num_cols = len(dict_of_noise["awgn"].keys()) -1 
      #Im not going to plot the name
      column_titles = [ col for col in dict_of_noise["awgn"].keys() if col != "input_dir_path"]
      
      fig = make_subplots(
          rows=num_rows,
          cols=num_cols,
          subplot_titles=column_titles,
          horizontal_spacing= 0.02,
          vertical_spacing= 0.05
      )
      
      for i, (noise_name, data) in enumerate(dict_of_noise.items()):
          
          row = i + 1
          depth_path = turn_relative_path_into_full(data["input_dir_path"], self.local_output)
          img_depth = self.load_image(os.path.join(depth_path, "depth_vis", os.path.basename(data["Normal Depth Map"])))
          img_error = self.load_image(turn_relative_path_into_full(data["Depth Map Error"], self.local_metrics))
          img_recon = self.load_image(turn_relative_path_into_full(data["3D Reconstruction"], self.local_metrics))
          
          fig.add_trace(go.Image(z=img_depth), row=row, col = 1)
          
          fig.add_trace(go.Image(z=img_error), row=row, col = 2)
          
          fig.add_trace(go.Image(z=img_recon), row=row, col = 3)
          
          fig.update_yaxes(title_text=noise_name.capitalize(), row=row, col=1)
      
      fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
      fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
      
      fig.update_layout(
      title_text="Evaluation of Noise in Images and DA3 Reconstructions",
      height=300 * num_rows, # ~300px per line
      width=1200,            # Total width
      margin=dict(l=50, r=20, t=80, b=20)
      )
      file_name = "evaluation_denoising.html"
      fig.write_html(file_name)
      
      print(f"File {file_name} was created with success!")
  
  """"
  What misses in the directories in this moment is:
    -Screenshot reconstructions images
    -Depth map errors image, need to call the function
  """
  def run_evaluation_pipeline(self,noise_documentation):
      
      full_path_clean = turn_relative_path_into_full(noise_documentation["clean"]["input_dir_path"], self.local_output)
      every_result = []
      for noise_type, informations in noise_documentation.items():
        
        if noise_type == "clean":
            continue
        
        full_path_noise = turn_relative_path_into_full(informations["input_dir_path"], self.local_output)
        
        self.generate_depth_error_image(full_path_clean, full_path_noise, noise_type)
        
        noise_result = self.calculate_3d_metrics(full_path_clean, full_path_noise, noise_type)
        
        if noise_result:
            every_result.append(noise_result)
            print(f"Metrics of {noise_type} saved..")
         
        noise_glb = glob.glob(os.path.join(full_path_noise,"*.glb"))[0] 
        
        reconstruction_path = turn_relative_path_into_full(noise_documentation[noise_type]["3D Reconstruction"], self.local_metrics)
        
        #Take a screenshot of the glb file
        self.generate_3d_screenshot(noise_glb, reconstruction_path)
      
      #Pick up de images and plot
      self.plot_page(noise_documentation)
    
      #SAVING METRICS IN A CSV--------------------------------------
      csv_path = os.path.join(self.local_metrics, "metrics_results.csv")
    
      with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        # Define columns in excel
        columns = ["Noise", "Mean Distance", "Std Deviation"]
        
        # Create a writer in excel
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        
        #Write excel header
        writer.writeheader()
    
        writer.writerows(every_result)
        
      print(f"\n Result table created with success in {csv_path}!")
      
    
      
    
      
      
      
      
      
    
    
      
    
      
  
    
  
      
      
      
  
