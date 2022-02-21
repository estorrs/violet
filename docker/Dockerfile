#FROM ubuntu:18.04
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN apt-get update && apt-get install -y vim wget curl git build-essential ffmpeg libsm6 libxext6

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN echo '12'
RUN git clone https://github.com/estorrs/violet.git 
RUN cd violet && git checkout development
RUN conda env create --file violet/env.yml 

# install violet
RUN /miniconda/envs/violet/bin/pip install -e violet

ENV PATH="/miniconda/envs/violet/bin:$PATH"

CMD /bin/bash
