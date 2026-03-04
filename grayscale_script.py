import cv2
import os
import glob
import numpy as np
DIR_FOR_OUTPUT = 'output_images'

images_extensions = ['*.png', '*.jpg']
video_extensions = ['*.mp4', '*.avi', '*.mkv']

images_found = []
videos_found = []

#Select specific stream
frames_dir = input("Select the directory you want to process: ")

parent_dir = os.path.dirname(os.path.normpath(frames_dir))
 
output_frames = os.path.join(parent_dir, DIR_FOR_OUTPUT)

# Verify if the output directory already exists 
if not os.path.exists(output_frames):
    os.makedirs(output_frames)

#Check if it is a video or a sequence of frames
for ext in images_extensions:
    images_found.extend(glob.glob(os.path.join(frames_dir,ext)))

for ext in video_extensions:
    videos_found.extend(glob.glob(os.path.join(frames_dir,ext)))
    
if len(images_found) > 0:
    
    print(f"-> Sequence of images detected: {len(images_found)} frames.")
    # List of frame paths ordered by the number
    sorted_paths = sorted(images_found)

    for index, f in enumerate(sorted_paths):
        
        #read the path directly
        cur_frame = cv2.imread(f)
        
        if cur_frame is None: 
            print(f"Couldn't read: {f}")
            continue
            
        # Apply the gray scale
        img_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        
        #Write the output in output_images folder
        output = os.path.join(output_frames, f'{index:04d}_output.png')
        cv2.imwrite(output, img_gray)

elif len(videos_found) > 0:
    video_path = videos_found[0]
    print(f"-> Video detected: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    index = 0
    
    while cap.isOpened():
        ret, cur_frame = cap.read()
        
        # No more frames, break the loop
        if not ret:
            break
            
        # Apply the gray scale
        img_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        
        #Write the output in output_images folder
        output = os.path.join(output_frames, f'{index:04d}_output.png')
        cv2.imwrite(output, img_gray)
        
        index += 1
        
    cap.release()

else:
    print("-> No images or videos found in the provided directory.")

print("Worked Fine!")