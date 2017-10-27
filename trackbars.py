import cv2

def nothing(x):
    pass

def setup():
	wind = cv2.namedWindow("Canny Edge Threshold Differences", cv2.WINDOW_AUTOSIZE)
	th1 = cv2.createTrackbar('Threshold 1', "Canny Edge Threshold Differences", 0, 100, nothing)
	th2 = cv2.createTrackbar('Threshold 2', "Canny Edge Threshold Differences", 0, 100, nothing)
	th3 = cv2.createTrackbar('Epsilon', "Canny Edge Threshold Differences", 0, 1000, nothing)
	gauss = cv2.createTrackbar('Gauss', "Canny Edge Threshold Differences", 0, 20, nothing)
	cv2.setTrackbarPos('Threshold 1','Canny Edge Threshold Differences', 65)
	cv2.setTrackbarPos('Threshold 2','Canny Edge Threshold Differences', 85)
	cv2.setTrackbarPos('Epsilon','Canny Edge Threshold Differences', 60)
	cv2.setTrackbarPos('Gauss','Canny Edge Threshold Differences', 1)

def getUpdate():
	threshold1 = cv2.getTrackbarPos('Threshold 1', 'Canny Edge Threshold Differences')
	threshold2 = cv2.getTrackbarPos('Threshold 2', 'Canny Edge Threshold Differences')
	epsilon = cv2.getTrackbarPos('Epsilon', 'Canny Edge Threshold Differences')
	sigma = cv2.getTrackbarPos('Gauss', 'Canny Edge Threshold Differences') * 2 + 1
	return [threshold1, threshold2, epsilon, sigma, sigma]