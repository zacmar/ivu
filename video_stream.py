ALL_CONTOURS = -1

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    threshold1 = 50
    threshold2 = 20
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    im2 = cv2.Canny(gray,threshold1,threshold2)
    im2, contours, hierarchy = cv2.findContours(im2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    indizes= []
    for cnt in contours:
        epsilon = 0.06*cv2.arcLength(cnt,True)
        indizes.append(cv2.approxPolyDP(cnt,epsilon,True))
    cont = [contour for contour in indizes if cv2.isContourConvex(contour) 
                                           and cv2.contourArea(contour) > 400
                                           and contour.shape[0] == 4]
    boundingRects = []
    for contour in cont:
        boundingRects.append(cv2.minAreaRect(contour))
    im2[::] = 0
    mask = im2
    means = []
    for boundingRect in boundingRects: 
        mask[::] = 0 # = im2
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(boundingRect)]), (255, 255, 255))
        means.append(cv2.mean(frame, mask))
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    for index, contour in enumerate(cont):
        cv2.drawContours(im2, [contour], ALL_CONTOURS, means[index], 5)
    cv2.imshow('frame', im2)
    cv2.imshow('grey', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
