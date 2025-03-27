FROM nvcr.io/nvidia/pytorch:25.01-py3

ARG DEBIAN_FRONTEND=noninteractive

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt update &&  \
#     apt -y install --no-install-recommends <your-package-list-here>

ARG USERNAME=runner
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt update \
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Cleanup
# RUN pip3 cache purge # cache is not enabled in the nvcr image
RUN apt autoremove -y
RUN apt clean

# Switch to user and install packages
USER $USERNAME
WORKDIR /SerpentVAE

# `--no-build-isolation` resolves `mamba-ssm`'s PyTorch version issue
# Refer to the "Installation" section at https://pypi.org/project/mamba-ssm/
COPY requirements.txt .
RUN cat requirements.txt | grep -Eo '(^[^#]+)' | xargs -n 1 pip install --no-build-isolation

# Copy the rest of the files
COPY . .

CMD [ "python3", "discrete_text_lightning_train.py" ]