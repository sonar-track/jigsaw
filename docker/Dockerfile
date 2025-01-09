FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID
ARG GROUP_ID
LABEL AUTHOR Phong Vo <phong.vodinh@gmail.com>
#RUN  sudo apt-key del 7fa2af80
RUN  apt update && apt install -y wget vim curl s3cmd ffmpeg build-essential locales fonts-powerline ffmpeg python3-tk firefox libturbojpeg apt-utils
# locales modules
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ARG USER=trainer
RUN useradd -m -u $USER_ID $USER
USER $USER_ID:$GROUP_ID
WORKDIR /home/$USER
ENV PATH="/usr/bin:/usr/local/bin:$HOME/.local/bin"

RUN pip install --upgrade pip

RUN pip3 install --upgrade \
    argparse namedlist scikit-learn scipy requests pandas h5py \
    matplotlib seaborn tqdm pillow scikit-image \
    opencv-python imageio notebook PyTurboJPEG tf2onnx \
    seaborn 

ENV PATH="${PATH}:/home/trainer/.local/bin"

CMD /bin/bash
