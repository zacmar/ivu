ALL_CONTOURS = -1

import cv2
import numpy as np

def nothing(x):
    pass

def setupTrackbars():
	wind = cv2.namedWindow("Canny Edge Threshold Differences", cv2.WINDOW_AUTOSIZE)
	th1 = cv2.createTrackbar('Threshold 1', "Canny Edge Threshold Differences", 0, 100, nothing)
	th2 = cv2.createTrackbar('Threshold 2', "Canny Edge Threshold Differences", 0, 100, nothing)
	th3 = cv2.createTrackbar('Epsilon', "Canny Edge Threshold Differences", 0, 1000, nothing)
	gauss = cv2.createTrackbar('Gauss', "Canny Edge Threshold Differences", 0, 20, nothing)
	cv2.setTrackbarPos('Threshold 1','Canny Edge Threshold Differences', 65)
	cv2.setTrackbarPos('Threshold 2','Canny Edge Threshold Differences', 85)
	cv2.setTrackbarPos('Epsilon','Canny Edge Threshold Differences', 60)
	cv2.setTrackbarPos('Gauss','Canny Edge Threshold Differences', 1)

def getTrackbarUpdate():
	threshold1 = cv2.getTrackbarPos('Threshold 1', 'Canny Edge Threshold Differences')
	threshold2 = cv2.getTrackbarPos('Threshold 2', 'Canny Edge Threshold Differences')
	epsilon = cv2.getTrackbarPos('Epsilon', 'Canny Edge Threshold Differences')
	sigma = cv2.getTrackbarPos('Gauss', 'Canny Edge Threshold Differences') * 2 + 1
	return [threshold1, threshold2, epsilon, sigma, sigma]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
mask = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

setupTrackbars()

while(True):
    ret, frame = cap.read()
    
    threshold1, threshold2, epsilon, sigma_color, sigma_space = getTrackbarUpdate()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_filtered = cv2.bilateralFilter(frame_gray, -1, sigma_color, sigma_space)
    frame_edges = cv2.Canny(frame_filtered, threshold1, threshold2)

    frame_edges, contours, hierarchy = cv2.findContours(frame_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_approx = [cv2.approxPolyDP(contour, epsilon / 1000 * cv2.arcLength(contour, True), True) 
    						for contour in contours]
    contours_filtered = [contour for contour in contours_approx 
    						if cv2.isContourConvex(contour)
                            and cv2.contourArea(contour, True) > 400
                            and contour.shape[0] == 4]
    bounding_rectangles = [cv2.minAreaRect(contour) for contour in contours_filtered]

    color_means = []
    for bounding_rectangle in bounding_rectangles:
        mask[::] = 0
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(bounding_rectangle)]), 255)
        color_means.append(cv2.mean(frame, mask))
    
    frame_composed = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)
    # frame_composed[::] = 0
    for index, contour in enumerate(contours_filtered):
        cv2.drawContours(frame_composed, [contour], ALL_CONTOURS, color_means[index], 5)
    cv2.imshow('Canny Edge Threshold Differences', frame_composed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()