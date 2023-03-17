### TODO: Make sure that the cuda driver's version matches (also below)
### TODO: Make sure you have your .dockerignorefile
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

### Use bash as the default shelll
RUN chsh -s /bin/bash
SHELL ["bash", "-c"]

### Add new Nvidia keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

### Install basics
RUN apt-get update && \
    apt-get install -y openssh-server sudo nano screen wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion unzip graphviz graphviz-dev && \
    apt-get clean

# Install miniconda
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    mkdir ~/.conda && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

### Install Environment
ENV envname understanding_decoding
RUN env "PATH=$PATH" conda update conda && \
    conda create --name $envname python=3.8

### Install a version of pytorch that is compatible with the installed cudatoolkit
RUN conda install -n $envname pytorch=1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
COPY requirements.yaml /tmp/requirements.yaml
RUN conda env update --name $envname --file /tmp/requirements.yaml --prune
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN conda run -n $envname pip install -r /tmp/pip_requirements.txt
RUN conda install -n $envname pygraphviz==1.9 -c conda-forge
RUN echo "conda activate $envname" >> ~/.bashrc && \
    conda run -n $envname python -m ipykernel install --user --name=$envname

### Setup-for installing apex
### TODO: Make sure that the cuda driver's version matches
ENV CUDAVER cuda-10.1
ENV PATH /usr/local/$CUDAVER/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
ENV CUDA_PATH /usr/local/$CUDAVER
ENV CUDA_ROOT /usr/local/$CUDAVER
ENV CUDA_HOME /usr/local/$CUDAVER
ENV CUDA_HOST_COMPILER /usr/bin/gcc

RUN mkdir /tmp/unique_for_apex
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN /opt/conda/envs/$envname/bin/pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /
