FROM dromni/nerfstudio:0.2.1

USER root

RUN apt-get update && \
    apt-get install curl 
    
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