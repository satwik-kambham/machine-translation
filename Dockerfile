# Set base image
FROM ubuntu:22.04

# Install required apt packages
RUN apt update
RUN apt install -y --no-install-recommends \
    git \
    ca-certificates \
    wget \
    curl

# Download and install mambaforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"
RUN bash Mambaforge-Linux-x86_64.sh -b
RUN rm -f Mambaforge-Linux-x86_64.sh

# Copy over environment yml specification
COPY environment.yml .

# Create and activate new environment from yml specification
ENV env_name=nlp
ENV env_bin=/root/mambaforge/envs/$env_name/bin
RUN ~/mambaforge/condabin/conda init
RUN ~/mambaforge/condabin/mamba env create --file environment.yml
RUN rm -f environment.yml

# Activate nlp environment
RUN echo "conda activate nlp" >> ~/.bashrc
