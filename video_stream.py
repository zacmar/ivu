import numpy as np
import cv2

cv2.namedWindow("Canny Edge Threshold Differences", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    threshold1 = 50
    threshold2 = 20
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    cv2.imshow('grey', gray)
    im2 = cv2.Canny(gray,threshold1,threshold2)
    im2, contours, hierarchy = cv2.findContours(im2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    convexContours = [contour for contour in contours if cv2.isContourConvex(contour) and cv2.contourArea(contour) > 1]
    indizes = []
    for cnt in contours:
        epsilon = 0.06*cv2.arcLength(cnt,True)
        indizes.append(cv2.approxPolyDP(cnt,epsilon,True))
    #cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
    cont = [contour for contour in indizes if cv2.isContourConvex(contour) and cv2.contourArea(contour) > 400]
    print(len(contours))
    im2[::] = 0
    cv2.drawContours(im2, cont, -1, (255,255,0), 5)
    cv2.imshow('frame', im2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
