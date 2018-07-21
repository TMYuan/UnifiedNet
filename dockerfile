FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Set anaconda path
ENV ANACONDA /opt/anaconda
ENV PATH $ANACONDA/bin:$PATH

# Download anaconda and install it
RUN apt-get update && apt-get install -y wget build-essential
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -P /tmp
RUN bash /tmp/Anaconda3-5.2.0-Linux-x86_64.sh -b -p $ANACONDA
RUN rm /tmp/Anaconda3-5.2.0-Linux-x86_64.sh

# Copy the config file into docker
RUN jupyter notebook --generate-config && ipython profile create
ADD jupyter_notebook_config.py  /root/.jupyter
ADD matplotlib_init.py /root/.ipython/profile_default/startup

# Install Pytorch
RUN conda install -y pytorch torchvision cuda90 -c pytorch

# Change workdir
WORKDIR /root/workspace

# Default command is to run a jupyter notebook at 0.0.0.0:8888 in headless mode
CMD ["bash"]