import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

# Use the local computer webcam (usually index 0)
cap = cv2.VideoCapture(0)

# If your built-in camera is not detected, try changing this to 1
# cap = cv2.VideoCapture(1)

# Optional: reduce the size for faster processing
cap.set(3, 320)
cap.set(4, 240)


def nothing(x):
    pass


cv2.namedWindow('HueComp')
cv2.namedWindow('SatComp')
cv2.namedWindow('ValComp')
cv2.namedWindow('closing')
cv2.namedWindow('tracking')

# HSV trackbars for the target color (#dfff4f)
cv2.createTrackbar('hmin', 'HueComp', 30, 179, nothing)
cv2.setTrackbarPos('hmin', 'HueComp', 19)
cv2.createTrackbar('hmax', 'HueComp', 40, 179, nothing)
cv2.setTrackbarPos('hmax', 'HueComp', 54)

cv2.createTrackbar('smin', 'SatComp', 120, 255, nothing)
cv2.setTrackbarPos('smin', 'SatComp', 69)
cv2.createTrackbar('smax', 'SatComp', 255, 255, nothing)
cv2.setTrackbarPos('smax', 'SatComp', 255)

cv2.createTrackbar('vmin', 'ValComp', 180, 255, nothing)
cv2.setTrackbarPos('vmin', 'ValComp', 203)
cv2.createTrackbar('vmax', 'ValComp', 255, 255, nothing)
cv2.setTrackbarPos('vmax', 'ValComp', 255)


while True:
    buzz = 0
    ret, frame = cap.read()

    if not ret:
        print("Unable to access webcam")
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    hmn = cv2.getTrackbarPos('hmin', 'HueComp')
    hmx = cv2.getTrackbarPos('hmax', 'HueComp')

    smn = cv2.getTrackbarPos('smin', 'SatComp')
    smx = cv2.getTrackbarPos('smax', 'SatComp')

    vmn = cv2.getTrackbarPos('vmin', 'ValComp')
    vmx = cv2.getTrackbarPos('vmax', 'ValComp')

    # Apply thresholding
    hthresh = cv2.inRange(np.array(hue), np.array(hmn), np.array(hmx))
    sthresh = cv2.inRange(np.array(sat), np.array(smn), np.array(smx))
    vthresh = cv2.inRange(np.array(val), np.array(vmn), np.array(vmx))

    # Combine channels
    tracking = cv2.bitwise_and(hthresh, cv2.bitwise_and(sthresh, vthresh))

    # Morphological filtering
    dilation = cv2.dilate(tracking, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    closing = cv2.GaussianBlur(closing, (5, 5), 0)

    # Detect circles
    circles = cv2.HoughCircles(
        closing,
        cv2.HOUGH_GRADIENT,
        2,
        120,
        param1=120,
        param2=50,
        minRadius=10,
        maxRadius=0,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if int(round(i[2])) < 30:
                cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), int(round(i[2])), (0, 255, 0), 5)
                cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), 2, (0, 255, 0), 10)
            elif int(round(i[2])) > 35:
                cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), int(round(i[2])), (0, 0, 255), 5)
                cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), 2, (0, 0, 255), 10)
                buzz = 1

    cv2.imshow('HueComp', hthresh)
    cv2.imshow('SatComp', sthresh)
    cv2.imshow('ValComp', vthresh)
    cv2.imshow('closing', closing)
    cv2.imshow('tracking', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
