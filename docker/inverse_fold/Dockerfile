FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y wget libxml2 git
RUN pip install fire 
RUN pip install rdkit-pypi==2022.3.1
RUN pip install git+https://github.com/facebookresearch/esm.git
# OpenFold and its remaining dependency
RUN pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
RUN pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

RUN pip install wandb

RUN pip install pytorch_lightning
RUN pip install networkx
RUN pip install matplotlib
RUN pip install einops
RUN apt update
WORKDIR /root

RUN apt install -y wget
RUN apt install -y vim
RUN apt install -y unzip

RUN pip install markupsafe==2.0.1

RUN conda install -y pyg -c pyg
RUN conda install -y -c conda-forge biopython
RUN conda install -y -c conda-forge biopandas

RUN pip install --upgrade pandas

RUN pip install transformers

RUN pip install torch-scatter
RUN pip install torch-sparse
RUN pip install torch-cluster
RUN pip install torch-spline-conv
RUN pip install torch-geometric

RUN pip install biotite
RUN pip install gpytorch 
RUN pip install botorch
# ENV WANDB_API_KEY=$YOUR_WANDB_API_KEY
WORKDIR /workspace

# Used to create public: nmaus/fold2:latest