FROM dromni/nerfstudio:0.3.0

USER root

RUN apt-get update && \
    apt-get install curl 
    
RUN apt-get install openssh-server -y supervisor   

# make .ssh directory and set permissions for user
RUN mkdir -p /home/user/.ssh && \
    chown -R user:root /home/user/.ssh && \
    chmod 700 /home/user/.ssh

# add symbolic link for python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# remove nerfstudio from /home/user/nerfstudio
RUN rm -rf /home/user/nerfstudio

USER user

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
RUN pip install -U mmcv-full
RUN pip install -U timm==0.4.12
RUN pip install -U mmdet==2.22.0
RUN pip install -U mmsegmentation==0.20.2