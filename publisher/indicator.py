import cv2
import numpy as np

from global_state_2 import get_angle_of_line

def indicator(frame):
    height, width, channel = frame.shape

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