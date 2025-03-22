#!/bin/sh

# Script taken and adapted from: https://github.com/psaboia/devcontainer-nvidia-base/tree/2ae1e1f12fd4873221a330ee31a6c92bd3c239c8

# update system
apt-get update && apt-get upgrade -y

# install Linux tools and Python 3
apt-get install software-properties-common wget curl git icu-devtools \
    python3-dev python3-pip python3-wheel python3-setuptools python3-packaging -y

# update CUDA Linux GPG repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb

# install cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
apt-get update
apt-get install libcudnn9=9.8.0.*-1+cuda12.8
apt-get install libcudnn9-dev=9.8.0.*-1+cuda12.8

# install Python packages
python3 -m pip install --upgrade pip
pip3 install torch  # causal-conv1d requires torch to be pre-installed
pip3 install -r requirements.txt

# clean up
pip3 cache purge
apt-get autoremove -y
apt-get clean