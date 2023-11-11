def detect_parking_sign(frame):
    height, width = frame.shape[:2]
    bottom_half = frame[(height // 2) - 40:, :]
    gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(1000))
    (corners, ids, rejected) = detector.detectMarkers(gray)
    if len(corners) > 0 and  ids == 10:
        return True
    return False
