#!/bin/bash
xhost +local:
docker run -it --rm --runtime nvidia --network host --device /dev/video0:/dev/video0:mrw -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix yolov5:jetpack-4-4-1 python3 detect.py --source 0
