import numpy as np
import glob
import os
import cv2
from helpers import create_clean_dirs
images_extensions = ['*.png', '*.jpg']
video_extensions = ['*.mp4', '*.avi', '*.mkv']

"""This class generates de noise and applies it to the clean video."""
class Noise:
  
  def __init__(self, seed, video_types, input_path, fps_wanted = 2):
    self.rng = np.random.default_rng(seed=seed)
    self.video_types = video_types
    self.input_path = input_path
    self.fps_wanted = fps_wanted
    

  def add_awgn_noise(self,cur_frame, rng, choosen_scale=25, choosen_loc=0):
    """Apply AWGN to the frames"""
    noise = rng.normal(loc=choosen_loc, scale=choosen_scale, size=cur_frame.shape)
            
    # Need to do this or else it could overflow
    img_noised = cur_frame.astype(np.float64) + noise 
    img_noised = np.clip(img_noised, 0, 255)
    
    return img_noised.astype(np.uint8) 

  #Apply salt and pepper with uniform distribution
  def add_salt_and_pepper(self,cur_frame, rng, a= 0.0, b= 1.0, salt_and_pepper_chance = 0.05):
    #We are going to add 5% salt and 5% chance of pepper
    
    noise = rng.uniform(a,b, cur_frame.shape)
    
    #Create a mask
    mask_pepper = noise <= (salt_and_pepper_chance/2)
    mask_salt = noise >= (1-(salt_and_pepper_chance/2))
    
    cur_frame[mask_pepper] = 0
    cur_frame[mask_salt] = 255
    
    return cur_frame

  def add_shot_noise(self,cur_frame,rng):
    
    #Can overflow
    img = cur_frame.astype(np.float64)
    
    noised_img = rng.poisson(img, img.shape)
    
    noised_img = np.clip(noised_img, 0, 255)
    
    return noised_img.astype(np.uint8)

  def add_speckle_noise(self,cur_frame, rng, choosen_loc=0, choosen_scale = 0.2):
    noise = rng.normal(loc=choosen_loc, scale=choosen_scale, size=cur_frame.shape)
            
    # Need to do this or else it could overflow
    img_noised = cur_frame.astype(np.float64)
    img_noised = img_noised + (img_noised * noise)  
    img_noised = np.clip(img_noised, 0, 255)
    
    return img_noised.astype(np.uint8)
  
  def process_single_frame(self,cur_frame, rng, frame_name, dir_writers):
    """Add noise, add grayscale and write them in video writer"""
   
    for video_type, dir_path in dir_writers.items():
      img_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
      
      match video_type:
        case "awgn":
          img_gray=self.add_awgn_noise(cur_frame=img_gray,rng=self.rng)
        case "salt_and_pepper":
          img_gray=self.add_salt_and_pepper(cur_frame=img_gray,rng=self.rng)
        case "shot_noise":
          img_gray = self.add_shot_noise(cur_frame=img_gray,rng=self.rng)
        case "speckle_noise":
          img_gray = self.add_speckle_noise(cur_frame=img_gray, rng=self.rng)
          
      cv2.imwrite(os.path.join(dir_path, frame_name), img_gray)
    
  #Main function
  """This function returns a dictionary for the current paths of the output directories."""
  def apply_noise(self):
    images_found = []
    videos_found = []

    for ext in images_extensions:
        images_found.extend(glob.glob(os.path.join(self.input_path, ext)))

    for ext in video_extensions:
        videos_found.extend(glob.glob(os.path.join(self.input_path, ext)))

    # Variáveis de setup
    is_video = False
    cap = None
    count = 0
    original_fps = 0
    # -Obtain properties of the frame
    if len(images_found) > 0:
        print(f"-> Sequence of images detected: {len(images_found)} frames.")
        images_found = sorted(images_found)
        
        # Read the first frame to obtain the size of them
        
    elif len(videos_found) > 0:
        video_path = videos_found[0]
        print(f"-> Video detected: {video_path}")
        is_video = True
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
    else:
        print("-> No images or videos found in the provided directory.")
        return #Return because it was no error

    dir_writers = {}
    #Use only the noise asked
    output_dir_path = None
    #Preparing video writers for each type of video
    for type in self.video_types:
      if type == "clean":
        output_dir_path = f"input_{type}"
        
      else:
        output_dir_path = f"input_noise_{type}"
      
      fullpath = os.path.join(self.input_path,output_dir_path)
      
      create_clean_dirs(fullpath)
      
      dir_writers[type] = fullpath
      
    if output_dir_path is None:
      print("No types registered")
    
    #PROCESSSING IMAGES 
    if not is_video:
        # Loop in images
        for frame_path in images_found:
            frame_name = os.path.basename(frame_path)
            cur_frame = cv2.imread(frame_path)
            if cur_frame is None: 
                print(f"Couldn't read: {frame_path}")
                continue
            
            self.process_single_frame(cur_frame, self.rng, frame_name, dir_writers)
            
    #PROCESSING VIDEOS
    else:
        if original_fps <= 0: 
            original_fps = 30
            
        self.frame_interval = int(round(original_fps / self.fps_wanted))
        if self.frame_interval < 1: 
            self.frame_interval = 1
        # Loop in videos
        while cap.isOpened():
            ret, cur_frame = cap.read()
            if not ret:
                break
              
            if count % self.frame_interval == 0:
              frame_name = f"frame_{count:05d}.jpg"
              self.process_single_frame(cur_frame, self.rng, frame_name, dir_writers)
            count += 1
        cap.release()
        
    
      
    
     

  
    