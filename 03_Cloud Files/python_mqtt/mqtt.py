import paho.mqtt.client as mqtt
import mysql.connector
import cv2
import boto3
import base64
import time
import numpy as np
import os
from pyzbar.pyzbar import decode

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)
	
def on_message(client, userdata, msg):
  try:
    print("message received in the cloud")
    img = base64.b64decode(msg.payload)
    f = np.frombuffer(img, dtype='uint8')
    img = cv2.imdecode(f, flags = 1)
    t = str(time.time())
    fileName = "image" + t + ".png"
    cv2.imwrite(fileName, img)

    #Write to S3
    s3 = boto3.client('s3',aws_access_key_id="AKIAZHIHFUS2LR5VQN7P", aws_secret_access_key="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")
    with open(fileName, "rb") as file_object:
        s3.upload_fileobj(file_object, "ericlundy87-image-bucket", fileName)
    print("message saved to s3")
    link = 'https://ericlundy87-image-bucket.s3.us-east-2.amazonaws.com/' + fileName

    #Read Barcode
    detectedBarcodes = decode(img)
    bc = None
    for barcode in detectedBarcodes:
        bc = barcode.data.decode("utf-8")
        print(bc)
        break
    val = (bc, link)
    mycursor.execute(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "record inserted.")
    os.remove(fileName)
  except:
    print("Unexpected error:", sys.exc_info()[0])

mydb = mysql.connector.connect(
    host="mariadb",
    user="eric",
    password="w251",
    database="inventory"
)

mycursor = mydb.cursor()

sql = "INSERT INTO SCANS (BARCODE, LINK) VALUES (%s, %s)"

LOCAL_MQTT_HOST="52.71.238.26"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="topic"

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()
