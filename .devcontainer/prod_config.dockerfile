FROM ubuntu:rolling
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND = noninteractive

RUN apt-get update && apt-get install -y \
  git \ 
  curl \ 
  wget \ 
  ffmpeg \ 
  htop \ 
  neovim \ 
  && apt-get clean \ 
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt . 

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES all 
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
