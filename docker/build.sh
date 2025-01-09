#!/usr/bin/env bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t sonar/sonar-ml:py38-tf-gpu .
