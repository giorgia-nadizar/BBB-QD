FROM mambaorg/micromamba:0.22.0 as conda

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONDA_DIR=/opt/conda
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV MAMBA_ROOT_PREFIX="/opt/conda"

COPY requirements.txt /tmp/requirements.txt
COPY environment.yml /tmp/environment.yml

USER root
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get -y install cmake
USER 1001

RUN micromamba create -y --file /tmp/environment.yml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as cuda-image
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install xvfb -y
RUN apt-get install libglfw3 -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install xorg-dev libglu1-mesa-dev -y
RUN apt-get install freeglut3-dev -y
RUN apt-get update && apt-get -y install cmake
ENV PATH=/opt/conda/envs/bbbqd/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH

ENV DISPLAY :0
# ENV LIBGL_ALWAYS_INDIRECT 1

ENV DISTRO ubuntu2004
ENV CPU_ARCH x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$CPU_ARCH/3bf863cc.pub

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN pip --no-cache-dir install jaxlib==0.3.15+cuda11.cudnn82 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && rm -rf /tmp/*
    
WORKDIR $APP_FOLDER

FROM cuda-image as run-image

RUN conda install -c conda-forge libstdcxx-ng -y

COPY evogymsrc evogymsrc
COPY setup.py ./
RUN python setup.py install

COPY examples/install_tester.py ./
CMD ["python", "./install_tester.py"]

# xvfb-run -a python ./video.py
#CMD ["python"]
