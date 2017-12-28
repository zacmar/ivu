import cv2
import numpy as np

def setupColorsFromUserInput():
	cap = cv2.VideoCapture(0)
	colors = []
	count = 0
	while(count < 6):
		ret, frame = cap.read()
		if cv2.waitKey(1) & 0xFF == ord('w'):
			roi = cv2.selectROI(frame, fromCenter=False)
			color = getMeanColorFromBoundingRect(frame, roi)
			colors.append(color)
			count += 1
			print(roi)
		else:
			cv2.imshow('select roi', frame)
	return colors

def getMeanColorFromBoundingRect(picture, bounding_rect):
    mask = cv2.cvtColor(picture.copy(), cv2.COLOR_RGB2GRAY)
    mask[::] = 0
    cv2.rectangle(mask, (bounding_rect[0], bounding_rect[1]), 
    					(bounding_rect[0] + bounding_rect[2], 
							bounding_rect[1] + bounding_rect[3]), 255, -1)
    return cv2.mean(picture, mask)