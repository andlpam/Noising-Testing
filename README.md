# Depth Anything 3: Noise Robustness Pipeline 🚁

This repository contains an automated pipeline to evaluate the robustness of the **Depth Anything 3 (DA3)** model against various types of image/video noise. 

The pipeline automatically injects noise into drone footage, runs 3D reconstruction, extracts normalized 2D Depth Maps, and calculates 3D geometric metrics using CloudCompare. 

**This pipeline features a Python Orchestrator** that seamlessly bridges GPU-heavy tasks (running inside a Docker container) and CPU-based metrics evaluation (running on your local host machine).

## 📋 Features
- **Noise Injection:** Generates variations of the input data (`awgn`, `salt_and_pepper`, `shot_noise`, `speckle_noise`).
- **3D Inference:** Uses DA3 (via Docker) to generate Point Clouds (`.glb`) and Depth Maps (`.npz`).
- **2D Evaluation:** Extracts and normalizes depth maps with a `plasma` colormap for visual comparison.
- **3D Evaluation:** Automates CloudCompare via CLI on the host machine (ICP + Cloud-to-Cloud Distance).
- **CSV Export:** Consolidates all 3D metrics into an easy-to-read Excel/CSV file.

## 🛠️ Prerequisites
1. **Python 3.x** installed on your host machine.
2. **Docker** installed with NVIDIA Container Toolkit (for GPU support).
3. **CloudCompare** installed on your host machine (required for the 3D metric extraction phase).

## 🚀 How to Run

### 1. Build the Docker Image
First, build the environment using the provided Dockerfile. You only need to do this once:
```bash
docker build -t da3-gpu .
```

### 2. Run the Pipeline
Use the main.py script to orchestrate the entire process. It will automatically start the Docker container for the heavy GPU inference, save the results, and then run CloudCompare locally on your host machine to extract the metrics.

#### Default Execution:

python main.py --local_data "C:\path\to\your\data"
(Note: Make sure your input videos are placed inside a videos_in folder within your --local_data directory).

#### Custom Execution (Example with flags):

python main.py --local_data "E:\drone_data" --fps 5 --noises clean awgn --cc_path "D:\Programs\CloudCompare\CloudCompare.exe"

⚙️ Available Flags:
--local_data (Required): Absolute path on your host machine to the base data folder where the pipeline will read inputs and save outputs.

--fps: Frames per second to extract (Default: 2)

--noises: Types of noise to apply. Choices: clean, awgn, salt_and_pepper, shot_noise, speckle_noise (Default: All)

--cc_path: Absolute path to your CloudCompare executable. (Default: C:\Program Files\CloudCompare\CloudCompare.exe)

📁 Output Structure
After the pipeline finishes, check the generated metrics_results folder inside your local data directory. You will find:

🖼️ comparation_[noise].png: Visual 2D representations of the depth maps (Clean vs Noisy) sharing the same global color scale.

📊 metrics_results.csv: A spreadsheet containing the Mean Distance and Standard Deviation for each noise type.

#### MIssing tasks:
- Put structural noise of all kinds
- Fix depth map errors
- Zoom Reconstructions
- Put metrics in the reconstruction image

📂 reconstructions_out folder: Containing the raw .glb point clouds and .npz depth tensors.


