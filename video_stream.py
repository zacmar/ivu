ALL_CONTOURS = -1

import cv2
import numpy as np
import copy
cap = cv2.VideoCapture(0)
def nothing(x):
    pass

wind = cv2.namedWindow("Canny Edge Threshold Differences", cv2.WINDOW_AUTOSIZE)
th1 = cv2.createTrackbar('Threshold 1', "Canny Edge Threshold Differences", 0, 100, nothing)
th2 = cv2.createTrackbar('Threshold 2', "Canny Edge Threshold Differences", 0, 100, nothing)
th3 = cv2.createTrackbar('Epsilon', "Canny Edge Threshold Differences", 0, 1000, nothing)
gauss = cv2.createTrackbar('Gauss', "Canny Edge Threshold Differences", 0, 20, nothing)
cv2.setTrackbarPos('Threshold 1','Canny Edge Threshold Differences', 65)
cv2.setTrackbarPos('Threshold 2','Canny Edge Threshold Differences', 85)
cv2.setTrackbarPos('Epsilon','Canny Edge Threshold Differences', 60)
cv2.setTrackbarPos('Gauss','Canny Edge Threshold Differences', 1)

while(True):
    ret, frame = cap.read()
    
    threshold1 = cv2.getTrackbarPos('Threshold 1','Canny Edge Threshold Differences')
    threshold2 = cv2.getTrackbarPos('Threshold 2','Canny Edge Threshold Differences')
    epstemp = cv2.getTrackbarPos('Epsilon','Canny Edge Threshold Differences')
    gaussval = cv2.getTrackbarPos('Gauss','Canny Edge Threshold Differences')
    gaussval = gaussval*2+1
    gray = cv2.bilateralFilter(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),9, gaussval, gaussval)
    im2 = cv2.Canny(gray,threshold1,threshold2)
    im2, contours, hierarchy = cv2.findContours(im2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    indizes= []
    for cnt in contours:
        epsilon = epstemp/1000*cv2.arcLength(cnt,True)
        indizes.append(cv2.approxPolyDP(cnt,epsilon,True))
        
    cont = [contour for contour in indizes if cv2.isContourConvex(contour)
                                           
                                           and contour.shape[0] == 4]
    boundingRects = []
    for contour in cont:
        boundingRects.append(cv2.minAreaRect(contour))
    #im2[::] = 0
    mask = im2.copy()
    means = []
    
    for boundingRect in boundingRects:
        mask[::] = 0 # = im2
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(boundingRect)]), (255, 255, 255))
        means.append(cv2.mean(frame, mask))
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)

    for index, contour in enumerate(cont):
        cv2.drawContours(im2, [contour], ALL_CONTOURS, means[index], 5)
    cv2.imshow('Canny Edge Threshold Differences', im2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
