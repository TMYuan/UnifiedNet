#!/bin/bash
nvidia-docker build -t pytorch:CUDA9
nvidia-docker run -i -t --name pytorch -v $(pwd):/root/workspace pytorch:CUDA9
