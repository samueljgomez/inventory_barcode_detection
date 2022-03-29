FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y python3-dev python3-pip  python3-opencv vim-tiny  libopencv-dev build-essential libzbar-dev
RUN pip3 install mysql-connector-python
RUN pip3 install paho-mqtt
RUN pip3 install boto3
RUN pip3 install pyzbar
COPY mqtt.py mqtt.py
CMD ["python3", "-u", "mqtt.py"]
