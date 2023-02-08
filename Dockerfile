FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN \
    # Update nvidia GPG key
    rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# preesed tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg; \
    echo "tzdata tzdata/Areas select Europe" >> /tmp/preseed.cfg; \
    echo "tzdata tzdata/Zones/Europe select Berlin" >> /tmp/preseed.cfg; \
    debconf-set-selections /tmp/preseed.cfg && \
    rm -f /etc/timezone /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata sudo nano git


# Install mambaforge
ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        bzip2 ca-certificates \
        libgl1 libglib2.0-0 \
        > /dev/null
RUN wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/22.11.1-2/Mambaforge-22.11.1-2-Linux-x86_64.sh -O /tmp/mambaforge.sh
RUN /bin/bash /tmp/mambaforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/mambaforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all
RUN mamba init bash

# LCDNet environment
COPY environment_lcdnet.yml /environment_lcdnet.yml
RUN mamba env create -f /environment_lcdnet.yml -n lcdnet

SHELL ["mamba", "run", "-n", "lcdnet", "/bin/bash", "-c"]

RUN git clone --depth 1 --branch v0.5.2 https://github.com/open-mmlab/OpenPCDet.git
WORKDIR /OpenPCDet

ENV CUDA_HOME=/usr/local/cuda-11.3/
ENV TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.5 8.0 8.6+PTX"
RUN python setup.py develop

WORKDIR /
RUN git clone --depth 1 https://github.com/robot-learning-freiburg/LCDNet.git
RUN mkdir /pretreined_models
COPY LCDNet-kitti360.tar /pretreined_models/LCDNet-kitti360.tar

# cleanup of files from setup
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN mamba clean -afy

