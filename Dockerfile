
#Torch needs to be between 2.0 and 2.4 otherwise some libraries cannot work.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

#Installing depedencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#Clonning the repository
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git .

#Updating their toml file. Opencv cannot be without a specific version otherwise it would be in conflict.
RUN sed -i 's/"opencv-python"/"opencv-python-headless<4.10"/g' pyproject.toml

RUN pip install "torchvision==0.17.2" "numpy<2.0" "xformers==0.0.25.post1"

RUN pip install -e ".[app]"

CMD ["/bin/bash"]

