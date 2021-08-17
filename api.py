import base64
import json
import os
import time
from PIL import Image
import requests
import cv2
import parameters

ENDPOINT = parameters.ENDPOINT

def send(timestamp, image, detections):
    #base64 image string from image
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)

    faces = []
    #create json from detections
    for i in range(len(detections)):
        c = detections[i][0]
        score = detections[i][1]
        x = int(detections[i][2])
        y = int(detections[i][3])
        w = int(detections[i][4])
        h = int(detections[i][5])
        mask_detected = False
        if c == 0:
            mask_detected = True
        face = {
            "bounds": {
                "x": x,
                "y": y,
                "width":w,
                "height":h
            },
            "mask_detected": mask_detected,
            "confidence": score
        }
        faces.append(face)

    jsonobject = {
        "timestamp": timestamp,
        "image": jpg_as_text.decode('utf-8'),
        #"image": 'image-data',
        "faces": faces
    }
    #app_json = json.dumps(jsonobject)
#    filename = str(timestamp) + '.txt'
#    with open(filename, 'w') as jsonfile:
#        json.dump(jsonobject, jsonfile)

    #post request
    x = requests.post(ENDPOINT, json=jsonobject)
#    print("sent frame with timestamp " + str(timestamp))
