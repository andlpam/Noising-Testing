import cv2
import os
import glob

OUTPUT_FILENAME = 'video_output.mp4'

images_extensions = ['*.png', '*.jpg']
video_extensions = ['*.mp4', '*.avi', '*.mkv']

images_found = []
videos_found = []

# Select specific stream
frames_dir = input("Select the directory you want to process: ")

parent_dir = os.path.dirname(os.path.normpath(frames_dir))
output_video_path = os.path.join(parent_dir, OUTPUT_FILENAME)

# Check if it is a video or a sequence of frames
for ext in images_extensions:
    images_found.extend(glob.glob(os.path.join(frames_dir, ext)))

for ext in video_extensions:
    videos_found.extend(glob.glob(os.path.join(frames_dir, ext)))
    
if len(images_found) > 0:
    print(f"-> Sequence of images detected: {len(images_found)} frames.")
    sorted_paths = sorted(images_found)

    # Obter as dimensões da primeira imagem para configurar o VideoWriter
    first_frame = cv2.imread(sorted_paths[0])
    height, width, _ = first_frame.shape
    
    # Configurar o codec e o VideoWriter (isColor=False porque estamos a usar tons de cinza)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_para_imagens = 30.0 # Podes alterar este valor consoante os FPS originais do teu drone
    out = cv2.VideoWriter(output_video_path, fourcc, fps_para_imagens, (width, height), isColor=False)

    for f in sorted_paths:
        cur_frame = cv2.imread(f)
        
        if cur_frame is None: 
            print(f"Couldn't read: {f}")
            continue
            
        # Apply the gray scale
        img_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        
        # Escrever a frame diretamente no ficheiro de vídeo
        out.write(img_gray)

    out.release() # Fechar e guardar o vídeo

elif len(videos_found) > 0:
    video_path = videos_found[0]
    print(f"-> Video detected: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Obter as propriedades originais do vídeo (FPS, largura e altura)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar o codec e o VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    while cap.isOpened():
        ret, cur_frame = cap.read()
        
        # No more frames, break the loop
        if not ret:
            break
            
        # Apply the gray scale
        img_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        
        # Escrever a frame no vídeo novo
        out.write(img_gray)
        
    cap.release()
    out.release() # Fechar e guardar o vídeo

else:
    print("-> No images or videos found in the provided directory.")

print(f"Worked Fine! Video saved at: {output_video_path}")