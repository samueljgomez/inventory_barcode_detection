
# Deploy Barcode Detection

This repository builds the docker image for object detection using Yolov5 on Nvidia Jetson platform.  Follow the steps below to build and deploy the docker image for barcode detection.  All operations should be on NVIDIA's Jetson NX - Jetpack 4.4.1

## Setup

This operation sets the default docker runtime to 'nvidia'.

```bash
sh setup.sh
```

## Build

This script builds the docker image named 'yolov5:jetpack-4-4-1'.  It should take a few hours. 

```bash
sh build.sh
```

## Run

This operation detects objects with camera connected to /dev/video0.

```bash
sh run.sh
```

## Run with your own weights

You can use your own weights(my-weights.pt), as follows:

```bash
mkdir -p /path/to/weights
cp my-weights.pt /path/to/weights
xhost +local:
docker run -it --rm \
           --runtime nvidia \
           --network host \
           --device /dev/video0:/dev/video0:mrw \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix/:/tmp/.X11-unix \
           -v /path/to/weights:/weights \
           yolov5:jetpack-4-4-1 python3 detect.py --source 0 --weights /weights/my-weights.pt
```
