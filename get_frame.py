import cv2
import numpy as np


def get_frame(info):
    rtsp = 'rtsp://' + info[0] + ':' + info[1] + '@192.168.10.'+info[2]+':554/Streaming/channels/' + info[3] + '01/'
    cap = cv2.VideoCapture(rtsp)
    ret, frame = cap.read()
    if not ret:
        print("Unable to succesfully retrieve the frame")
    return frame
