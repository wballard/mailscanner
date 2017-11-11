#base image provides CUDA support on Ubuntu 16.04
FROM nvidia/cuda:8.0-cudnn6-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ENV NB_USER keras
ENV NB_UID 1000

# package updates to support conda
RUN apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz

# force conda into the root path
RUN echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# setting up a non-root user to run conda
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src
USER keras

# add on conda python and make sure it is in the path
RUN mkdir -p $CONDA_DIR && \
    wget --quiet --output-document=$CONDA_DIR/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN /bin/bash $CONDA_DIR/miniconda.sh -f -b -p $CONDA_DIR && \
    rm $CONDA_DIR/miniconda.sh

# requirement files to get needed packages
COPY ./*.txt /mailscanner/
RUN conda install --file /mailscanner/conda-requirements.txt
RUN	pip install --requirement /mailscanner/requirements.txt

# preload vectoria word vectors
RUN python -c 'import vectoria; vectoria.CharacterTrigramEmbedding()'