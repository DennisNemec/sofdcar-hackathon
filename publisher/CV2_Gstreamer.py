#!/usr/bin/python
import sys
import gi
gi.require_version('GLib', '2.0')
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import cv2
import time
import CameraData
import globals as G

import numpy as np

def get_angle_of_line(l, alpha):
    x1 = l[0]
    y1 = l[1]
    x2 = l[2]
    y2 = l[3]

    angle = 0
    try:
        # angle = np.rad2deg(np.arctan((y2-y1)/(x2-x1)))
        angle = np.rad2deg(np.arctan((x2-x1)/(y2-y1)))
    except ZeroDivisionError:
        pass
    return angle

def indicator(frame):
    height, width, channel = frame.shape

        # Cut the image in half
    height_cutoff = height // 2
    s1 = frame[:height_cutoff, :]
    frame = frame[height_cutoff:,:]

    x_mid = int(width / 2)
    y_upper_mid = 0
    y_down_mid = height

    alpha = 3 # Contrast control (1.0-3.0)
    beta = 5 # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    ret, thresh1 = cv2.threshold(frame, 230, 255, cv2.THRESH_BINARY) 

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    #cv2.line(closing, (x1Mean, y1Mean), (x2Mean, y2Mean), (255,0,0), 3)
    #cv2.line(closing, (x_mid, y_upper_mid), (x_mid, height), (0,255,0), 3)

    kernel = np.ones((5,5),np.float32)/25
    closing = cv2.filter2D(closing,-1,kernel)

    ret, thresh = cv2.threshold(closing, 100, 255, cv2.THRESH_BINARY)


    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(closing, 2, np.pi/180, 50,minLineLength=50, maxLineGap=100)

    x1Mean = []
    x2Mean = []

    y1Mean = []
    y2Mean = []

    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    for line in lines:
        x1, y1, x2, y2 = line[0]                                                                                                                                
        x1Mean.append(x1)
        x2Mean.append(x2)
        y1Mean.append(y1)
        y2Mean.append(y2)

    x1Mean = int(np.floor(np.mean(x1Mean)))
    x2Mean = int(np.floor(np.mean(x2Mean)))
    y1Mean = int(np.floor(np.mean(y1Mean)))
    y2Mean = int(np.floor(np.mean(y2Mean)))

    cv2.line(thresh, (x1Mean, y1Mean), (x2Mean, y2Mean), (255, 0, 0), 5)

    angle = get_angle_of_line([x1Mean,y1Mean,x2Mean,y2Mean], 100)

    print("Angle: {}°".format(int(np.floor(angle))))

    cv2.imshow("Frame", thresh)

def detect_parking_sign(frame):
    print("here detetct")
    print("-----------------------------------------")
    height, width = frame.shape[:2]
    print("Height: {}".format(height))
    print("Width: {}".format(width))
    
    bottom_half = frame[(height // 2) - 50:, :]
    gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(1000))
    (corners, ids, rejected) = detector.detectMarkers(gray)

    if len(corners) > 0:
        pts = np.array([[751, 645], [838, 645], [831, 560], [747, 559]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 0, 0), 5)

        int_corners = np.int0(corners)
        print(int_corners)

        cv2.polylines(frame, int_corners, True, (0, 255, 0), 5)
        cv2.imshow("PARKING", frame)
        
        print("found corner")
        """if ids == 10:
            print("found 10")
            return True"""
    return False


Gst.init(None)
G.IP_ADDRESS="0.0.0.0"
vehicle_camera=CameraData.GstUdpCamera(9000)
vehicle_camera.play()
while True:
    if not vehicle_camera.new_imgAvaiable:
        print("no data")
        time.sleep(5)
        print("sleeping for 5 seconds")
    img = vehicle_camera.new_imgData
    img_rgb=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detect_parking_sign(img_rgb)
    indicator(img)
    cv2.imshow("camera", img_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

