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

### 2. Run the Docker Image with the following flags:

#### ⚙️ Standard Flags (Adjust these for your experiment):
* **`--fps`**: Frames per second to extract *(Default: `2`)*
* **`--noises`**: Types of noise to apply. Choices: `clean`, `awgn`, `salt_and_pepper`, `shot_noise`, `speckle_noise` *(Default: All)*

#### 🔧 Advanced Flags (Internal Container Paths):
*(Note: If you are using the default Docker `-v` volume mount to `/app/data`, you do not need to change these).*
* **`--input`**: Path to input videos inside the container *(Default: `/app/data/videos_in`)*
* **`--output`**: Path to save GLB/NPZ files inside the container *(Default: `/app/data/metrics_results`)*
* **`--cc_path`**: Path to the CloudCompare executable.

Default example(Applicable to windows machines):
docker run --gpus all -it --rm -v /path/to/your/local/data:/app/data da3-gpu

Custom Execution(Recommended):
docker run --gpus all -it --rm -v /path/to/your/local/data:/app/data da3-gpu python run_reconstructions.py --fps 5 --noises clean awgn --cc_path /path/to/your/cloudCompare/executable --output /path/to/send/reconstructions/output --metrics /path/to/your/metrics/folder

📁 Output Structure
After the pipeline finishes, check your mapped metrics_results folder. You will find:

🖼️ comparation_[noise].png: Visual 2D representations of the depth maps (Clean vs Noisy) sharing the same global color scale.

📊 metrics_results.csv: A spreadsheet containing the Mean Distance and Standard Deviation for each noise type.

📂 Folders per noise: Containing the raw .glb point clouds and .npz depth tensors.
