# Dev Container Setup

The dev container requires the following drivers on your
system before it sets the rest of the environment up for
you:

1. CUDA $\geq$ 12.4
2. Latest Nvidia Drivers

## Troubleshooting

If Docker outputs the following error log:

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

It means that Docker cannot find the NVIDIA cards on your device due to a lack of
drivers installed. Run the following commands on your **host** machine:

```bash
# 1. Configure the repository:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update

# 2. Install the NVIDIA Container Toolkit packages:
sudo apt-get install -y nvidia-container-toolkit

# 3: Configure the container runtime by using the nvidia-ctk command:
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart the Docker daemon:
sudo systemctl restart docker
```

Refer to the StackOverflow post [here](https://stackoverflow.com/a/77269071) for more info.

## Files

The files in this folder are currently taken and
adapted from [here](https://github.com/psaboia/devcontainer-nvidia-base/tree/2ae1e1f12fd4873221a330ee31a6c92bd3c239c8).

- `devcontainer.json` - Configuration files to set the
  devcontainer up. Currently it only contains the setup
  script to run, and the VSCode extensions to be installed
  in the devcontainer
- `setup-env.sh` - The core of the devcontainer. Used to
  install all packages and set the environment up for
  the devcontainer
