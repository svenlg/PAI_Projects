# FROM python:3.8.5-slim
# ADD ./requirements.txt /requirements.txt
# RUN pip install -r /requirements.txt
# WORKDIR /code
# ADD * /code/
# ADD pytransform /code/pytransform
# WORKDIR /code
# CMD python -u checker_client.py

FROM mambaorg/micromamba:latest
USER root
RUN apt-get update && apt-get install -y gcc
# apt-get update &&
RUN apt install -y libssl-dev
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install pyarmor==6.7.4
RUN pip install Gpy==1.9.9
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
CMD python -u checker_client.py