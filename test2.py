import cv2
import numpy as np
cap = cv2.VideoCapture("./publisher/video.mp4")
cap.set(3, 160)
cap.set(4, 120)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    height, width, channel = frame.shape

    x_mid = int(width / 2)
    y_upper_mid = 0
    y_down_mid = height

    alpha = 1 # Contrast control (1.0-3.0)
    beta = -500 # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    ret, thresh1 = cv2.threshold(frame, 254, 255, cv2.THRESH_BINARY) 

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(closing, 2, np.pi/180, 50,minLineLength=50, maxLineGap=100)
    closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)

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

    cv2.line(closing, (x1Mean, y1Mean), (x2Mean, y2Mean), (255,0,0), 3)
    cv2.line(closing, (x_mid, y_upper_mid), (x_mid, height), (0,255,0), 3)

    cv2.imshow("Frame", closing)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
                break
    
cap.release()
cv2.destroyAllWindows()