import cv2
import numpy as np

cap = cv2.VideoCapture("./video.mp4")
cap.set(3, 160)
cap.set(4, 120)
while True:

    ret, img = cap.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    #converted = convert_hls(img)
    image = cv2.cvtColor(blur_gray,cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0,   100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = img.copy()
    cv2.imshow("mask",mask) 



    # ret, frame = cap.read()
    # # grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    # # grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    # # grad = np.sqrt(grad_x**2 + grad_y**2)
    # # grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

    # alpha = 1 # Contrast control (1.0-3.0)
    # beta = -300 # Brightness control (0-100)

    # frame[:,:,2] = np.zeros([frame.shape[0], frame.shape[1]])

    # # blur = cv2.GaussianBlur(frame, (0,0), sigmaX=1, sigmaY=1)
    # # frame = cv2.divide(frame, blur, scale=255)

    # adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # # blur = cv2.GaussianBlur(adjusted, (0,0), sigmaX=5, sigmaY=5)
    # # frame = cv2.divide(adjusted, blur, scale=255)
    # # imagem = cv2.bitwise_not(adjusted)
    # # imagem = cv2.convertScaleAbs(frame, alpha=10, beta=0)

    
    # ret, thresh1 = cv2.threshold(frame, 254, 255, cv2.THRESH_BINARY) 

    # kernel = np.ones((5,5), np.uint8)
    # opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # # ret, thresh2 = cv2.threshold(frame, 120, 0, cv2.THRESH_BINARY_INV) 

    # # adjusted = adjusted[np.where((adjusted > 0.55) & (adjusted < 0.95))] = 0

    # cv2.imshow("Frame", closing)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
                break


            
cap.release()
cv2.destroyAllWindows()