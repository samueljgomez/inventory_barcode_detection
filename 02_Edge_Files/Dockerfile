FROM sagotechnology/yolov5

# tested on Jetson NX 4.4.1

WORKDIR /usr/src/app/yolov5

COPY detect_custom.py ./

RUN apt update && apt install -y mosquitto-clients
RUN pip3 install -U paho-mqtt

