def drive_straight(image):
    height, width = image.shape[:2]
    third_width = width // 3
    center_third = image[:, third_width:2 * third_width]
    height = center_third.shape[0]
    third_height = height // 3
    frame = center_third[2 * third_height:, :]
    ret, thresh1 = cv2.threshold(frame, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(closing, 2, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
    closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    if lines is None:
        return False
    return True
