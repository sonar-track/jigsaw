#!/usr/bin/env bash
docker run --name $1 --init --rm -it \
    --hostname workstation \
    -e DISPLAY="$${DISPLAY}" -e "TERM=xterm-256color" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/dp/dev:/home/trainer/dev \
    --gpus all \
    sonar/sonar-ml:py38-tf-gpu bash
