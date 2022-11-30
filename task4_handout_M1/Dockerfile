FROM mambaorg/micromamba:latest
USER root
RUN apt-get update && apt-get install -y gcc && apt-get install -y g++ 
RUN apt install -y libssl-dev && apt-get install -y cmake zlib1g-dev
RUN apt install -y xvfb python3-opengl ffmpeg 
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN pip install pyarmor==6.7.4
RUN pip install Cython

RUN pip install gym==0.21.0
RUN pip install Box2D==2.3.2
RUN pip install box2d-kengz
RUN pip install torch==1.8.0
RUN pip install moviepy
RUN pip install pyopengl
RUN pip install pyglet==1.5.11


WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
CMD xvfb-run -s "-screen 0 1400x900x24" python -u checker_client.py