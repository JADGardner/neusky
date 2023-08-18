### How to setup Nerfstudio on VarGPUs

Normally you would just use a conda environment. But currently due to issues with the version of glibc on the servers that IT support say cannot be changed we have to use a Singularity container.

This is sort of like a Docker container (which is sort of like a virtual machine but not) but that has no sudo privilages on the host machine.

You first need to build / obtain from one I made early a singularity .sif file (the container)

### To build a container

1. Install apptainer: https://apptainer.org/docs/admin/main/installation.html
2. Write a definition file, you might need to update nerfstudio version at the top:

```shell
BootStrap: docker
From: dromni/nerfstudio:0.3.1
Stage: spython-base

%environment
  export PYTHONPATH=$PYTHONPATH:/home/user/.local/lib/python3.10/site-packages

%post
  touch /etc/localtime
  touch /usr/bin/nvidia-smi
  touch /usr/bin/nvidia-debugdump
  touch /usr/bin/nvidia-persistenced
  touch /usr/bin/nvidia-cuda-mps-control
  touch /usr/bin/nvidia-cuda-mps-server
  mkdir -p /run/nvidia-persistenced
  touch /run/nvidia-persistenced/socket

  apt update
  apt install -y curl
  apt install -y python3.10-venv

  su - user

  su -

  chmod -R 777 /home/user/

  su - user
```

3. Build .sif file on your own machine:

```shell
sudo apptainer build output.sif input.def
```

4. Move .sif file to server

### Using one I built earlier

1. Copy it from the shared folder
