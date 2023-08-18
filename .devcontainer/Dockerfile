FROM dromni/nerfstudio:0.3.0

USER root

RUN apt-get update && \
    apt-get install curl

RUN apt-get install libopenexr-dev libilmbase-dev
    
RUN apt-get install openssh-server -y supervisor   

# make .ssh directory and set permissions for user
RUN mkdir -p /home/user/.ssh && \
    chown -R user:root /home/user/.ssh && \
    chmod 700 /home/user/.ssh

# remove nerfstudio from /home/user/nerfstudio
RUN rm -rf /home/user/nerfstudio

USER 1000

# update pip 
RUN pip install -U pylint
RUN pip install -U black
RUN pip install -U yacs
RUN pip install -U icosphere
RUN pip install -U tqdm
RUN pip install -U wget
RUN pip install -U trimesh
RUN pip install -U ipykernel
RUN pip install -U jupyterlab
RUN pip install -U ipywidgets
RUN pip install -U torch-tb-profiler
RUN pip install -U pytest
RUN pip install -U pip-system-certs
RUN pip install -U wandb --upgrade
RUN pip install -U open_clip_torch
RUN pip install -U timm
RUN pip install -U ftfy
RUN pip install -U regex
RUN pip install -U torchtyping
RUN pip install -U clip@git+https://github.com/openai/CLIP.git
RUN pip install -U --upgrade jupyter-server
RUN pip install -U openmim
RUN pip install -U openexr==1.3.8
RUN pip install -U pyexr
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"
RUN pip install "mmsegmentation>=1.0.0"
RUN pip install "mmdet>=3.0.0rc4"