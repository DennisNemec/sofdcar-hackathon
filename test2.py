import cv2
import numpy as np
cap = cv2.VideoCapture("./publisher/video.mp4")
cap.set(3, 160)
cap.set(4, 120)
while True:
    ret, frame = cap.read()
    # grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    # grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    # grad = np.sqrt(grad_x**2 + grad_y**2)
    # grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

    alpha = 1 # Contrast control (1.0-3.0)
    beta = -500 # Brightness control (0-100)

    

    # blur = cv2.GaussianBlur(frame, (0,0), sigmaX=1, sigmaY=1)
    # frame = cv2.divide(frame, blur, scale=255)

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # blur = cv2.GaussianBlur(adjusted, (0,0), sigmaX=5, sigmaY=5)
    # frame = cv2.divide(adjusted, blur, scale=255)
    # imagem = cv2.bitwise_not(adjusted)
    # imagem = cv2.convertScaleAbs(frame, alpha=10, beta=0)

    
    ret, thresh1 = cv2.threshold(frame, 254, 255, cv2.THRESH_BINARY) 

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # ret, thresh2 = cv2.threshold(frame, 120, 0, cv2.THRESH_BINARY_INV) 

    # adjusted = adjusted[np.where((adjusted > 0.55) & (adjusted < 0.95))] = 0

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


    cv2.imshow("Frame", closing)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
                break
    
cap.release()
cv2.destroyAllWindows()