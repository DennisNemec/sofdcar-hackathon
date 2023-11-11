from contextlib import closing
import cv2
import numpy as np


def get_closing_and_lines(img):
    alpha = 1 # Contrast control (1.0-3.0)
    beta = -500 # Brightness control (0-100)

    ret, thresh1 = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY) 

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(closing, 2, np.pi/180, 50,minLineLength=50, maxLineGap=100)
    closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    return (closing, lines) # doing two returns otherwise require more functions

def get_avg_line(lines):
    x1Mean = []
    x2Mean = []

    y1Mean = []
    y2Mean = []

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
    return x1Mean, x2Mean, y1Mean, y2Mean

def get_angle_of_line(l, alpha):
    x1 = l[0]
    y1 = l[1]
    x2 = l[2]
    y2 = l[3]

    angle = 0
    try:
        # angle = np.rad2deg(np.arctan((y2-y1)/(x2-x1)))
        angle = -1*np.rad2deg(np.arctan((x2-x1)/(y2-y1)))
    except ZeroDivisionError:
        pass
    return angle

def get_offset(l, width_img, alpha):
    # get distance of endpoint closest to center vertical
    center = width_img//2
    x1 = l[0]
    y1 = l[1]
    x2 = l[2]
    y2 = l[3]

    if abs(x1-center) < abs(x2-center):
        return alpha*(x1-center)
    else:
        return alpha*(x2-center)


def determine_stearing(angle, offset):
    if abs(angle) > 35:
        sign = 1 if angle > 0 else -1
        angle = sign*35

    return angle/35

# get lines in image

cap = cv2.VideoCapture("./video.mp4")
cap.set(3, 160)
cap.set(4, 120)

previous_lines = []

def stay_on_line(img):
    length = len(img)
    width = len(img[0])

    my_line = [width//2,0, width//2,length]

    closing, lines = get_closing_and_lines(img)

    x1Mean, x2Mean, y1Mean, y2Mean = get_avg_line(lines)

    cv2.line(closing, (x1Mean, y1Mean), (x2Mean, y2Mean), (255,0,0), 3)

    angle = get_angle_of_line([x1Mean,y1Mean,x2Mean,y2Mean], 100)
    # print(determine_stearing(angle, 0))

    # print(get_offset([x1Mean,y1Mean,x2Mean,y2Mean], width, 100))

    cv2.line(closing, (my_line[0], my_line[1]), (my_line[2], my_line[3]), (0,255,0),3)

    # cv2.imshow("Frame", closing)
    return determine_stearing(angle, 0)

if __name__ == "__main__":
    while True:
        ret, img = cap.read()
        length = len(img)  # i don't want this in while loop either
        width = len(img[0])


        stay_on_line(img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
                    break