FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND = noninteractive

RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-dev \
  python3-venv \
  python3-setuptools \
  git \ 
  curl \ 
  wget \ 
  ffmpeg \ 
  htop \ 
  neovim \ 
  && apt-get clean \ 
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
  ln -sf /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt . 

RUN pip3 install --no-cache-dir --upgrade pip && \ 
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
  pip3 install --no-cache-dir -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES all 
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility