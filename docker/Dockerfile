FROM ubuntu:18.04

RUN apt-get update && apt-get install -y vim wget curl git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN echo "world"
RUN git clone https://github.com/estorrs/violet.git 
RUN cd violet && git checkout development
RUN conda env create --file /violet/env.yml 

ENV PATH="/miniconda/envs/violet/bin:$PATH"

CMD /bin/bash
