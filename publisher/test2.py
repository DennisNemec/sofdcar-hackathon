from math import dist
import cv2
import numpy as np

cap = cv2.VideoCapture("./video.mp4")
cap.set(3, 160)
cap.set(4, 120)

previous_lines = []

while True:

    ret, img = cap.read()
    length = len(img)
    width = len(img[0])
    print(f"LENGTH: {length}")
    # print(len(img))
    # print(len(img[0]))
    # exit(0)

    black_image = np.zeros((len(img), len(img[0]), 1))
    # print(len(black_image))
    # exit()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # low_threshold = 200
    # high_threshold = 0
    low_threshold = 250
    high_threshold = 250
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    new_lines = []
    x1_mean = []
    x2_mean = []
    y1_mean = []
    y2_mean = []

    my_line = [width//2,0, width//2,length]


    def sorted_dot(l):
        x1,y1,x2,y2 = l[0]
        x3 = x2-x1
        y3 = y2-y1
        
        my_line_x = my_line[2]-my_line[0]
        my_line_y = my_line[3]-my_line[1]

        return (my_line_x*x3)+(my_line_y*y3)




    # sort lines according to which are closest to the center
    my_line = sorted(lines, key=sorted_dot, reverse=True)[0][0]







    # get line length
    for i in range(len(lines)):
        x1,y1,x2,y2 = lines[i][0]
        x1_mean.append(x1)
        x2_mean.append(x2)
        y1_mean.append(y1)
        y2_mean.append(y2)
        # print(line)
        distance = ((x2-x1)**2+(y2-y1)**2)**0.5
        print(f"DISTANCE: {distance}")
        if distance > 165:
            x1_mean.append(x1)
            x2_mean.append(x2)
            y1_mean.append(y1)
            y2_mean.append(y2)
            new_lines.append(lines[i])
        else:
            print("removed")

    x1_mean = int(np.mean(x1_mean))
    x2_mean = int(np.mean(x2_mean))
    y1_mean = int(np.mean(y1_mean))
    y2_mean = int(np.mean(y2_mean))


    lines = new_lines

    # cv2.line(line_image, (x1_mean, y1_mean), (x2_mean, y2_mean), (255,0,0),5)


    # my_line = [width//2,0, width//2,length]

    cv2.line(line_image, (my_line[0], my_line[1]), (my_line[2], my_line[3]), (255,0,0),5)



    # for line in lines:
    #     print(line)
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    #     # break

    # for line in previous_lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    previous_lines = lines

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imshow("mask",line_image) 

    if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
                break


            
cap.release()
cv2.destroyAllWindows()