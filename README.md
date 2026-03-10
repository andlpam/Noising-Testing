# Depth Anything 3: Noise Robustness Pipeline 🚁

This repository contains an automated pipeline to evaluate the robustness of the **Depth Anything 3 (DA3)** model against various types of image/video noise. 

The pipeline automatically injects noise into drone footage, runs 3D reconstruction, extracts normalized 2D Depth Maps, and calculates 3D geometric metrics using CloudCompare.

## 📋 Features
- **Noise Injection:** Generates variations of the input data (`awgn`, `salt_and_pepper`, `shot_noise`, `speckle_noise`).
- **3D Inference:** Uses DA3 to generate Point Clouds (`.glb`) and Depth Maps (`.npz`).
- **2D Evaluation:** Extracts and normalizes depth maps with a `plasma` colormap for visual comparison.
- **3D Evaluation:** Automates CloudCompare via CLI (ICP + Cloud-to-Cloud Distance) to extract Mean Distance and Standard Deviation.
- **CSV Export:** Consolidates all 3D metrics into an easy-to-read Excel/CSV file.

## 🛠️ Prerequisites
1. **Docker** installed with NVIDIA Container Toolkit (for GPU support).
2. **CloudCompare** installed (required for the 3D metric extraction phase).

## 🚀 How to Run

### 1. Build the Docker Image
First, build the environment using the provided Dockerfile:

docker build -t da3-gpu .

### 2. Run the Docker Image with the following flags:

--input: Path to input videos (Default: /app/data/videos_in)

--output: Path to save GLB/NPZ files (Default: /app/data/metrics_results)

--metrics: Path to save PNG and CSV metrics (Default: /app/data/metrics_results)

--fps: Frames per second to extract (Default: 2)

--noises: Types of noise to apply. Choices: clean, awgn, salt_and_pepper, shot_noise, speckle_noise. (Default: All of them)

--cc_path: Path to the CloudCompare executable. (Different for every operating system.)

Default example(Applicable to windows machines):
docker run --gpus all -it --rm -v /path/to/your/local/data:/app/data da3-gpu

Custom Execution(Recommended):
docker run --gpus all -it --rm -v /path/to/your/local/data:/app/data da3-gpu python run_reconstructions.py --fps 5 --noises clean awgn --cc_path /path/to/your/cloudCompare/executable --output /path/to/send/reconstructions/output --metrics /path/to/your/metrics/folder

📁 Output Structure
After the pipeline finishes, check your mapped metrics_results folder. You will find:

🖼️ comparation_[noise].png: Visual 2D representations of the depth maps (Clean vs Noisy) sharing the same global color scale.

📊 metrics_results.csv: A spreadsheet containing the Mean Distance and Standard Deviation for each noise type.

📂 Folders per noise: Containing the raw .glb point clouds and .npz depth tensors.
