import cv2
import numpy as np

ALL_CONTOURS = -1

import cv2

def th1Chng(x):
    pass

def th2Chng(x):
    pass

def epsChng(x):
    pass

wind = cv2.namedWindow("Canny Edge Threshold Differences", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)
th1 = cv2.createTrackbar('Threshold 1', "Canny Edge Threshold Differences", 0, 100, th1Chng)
th2 = cv2.createTrackbar('Threshold 2', "Canny Edge Threshold Differences", 0, 100, th2Chng)
th3 = cv2.createTrackbar('Epsilon', "Canny Edge Threshold Differences", 0, 1000, epsChng)
cv2.setTrackbarPos('Threshold 1','Canny Edge Threshold Differences', 65)
cv2.setTrackbarPos('Threshold 2','Canny Edge Threshold Differences', 85)
cv2.setTrackbarPos('Epsilon','Canny Edge Threshold Differences', 60)
while(True):
    threshold1 = cv2.getTrackbarPos('Threshold 1','Canny Edge Threshold Differences')
    threshold2 = cv2.getTrackbarPos('Threshold 2','Canny Edge Threshold Differences')
    epstemp = cv2.getTrackbarPos('Epsilon','Canny Edge Threshold Differences')
    ret, frame = cap.read()

    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    im2 = cv2.Canny(gray,threshold1,threshold2)

    im2, contours, hierarchy = cv2.findContours(im2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


    #convexContours = [contour for contour in contours if cv2.isContourConvex(contour) and cv2.contourArea(contour) > 1]

    indizes = []
    for cnt in contours:
        epsilon = epstemp/1000*cv2.arcLength(cnt,True)
        indizes.append(cv2.approxPolyDP(cnt,epsilon,True))

    #cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)

    cont = [contour for contour in indizes if cv2.isContourConvex(contour)
            and cv2.contourArea(contour) > 400
            and contour.shape[0] == 4]

    #im2[::] = 0

    test = cv2.drawContours(im2, cont, ALL_CONTOURS, (255,255,0), 5)
    #videostreams = np.hstack((gray,test))
    #completewindow = np.vstack((videostreams, th1))
    cv2.imshow("Canny Edge Threshold Differences", test)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
