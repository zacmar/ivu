ALL_CONTOURS = -1

import cv2
import numpy as np
import config, trackbars

def houghLineTransform(input_image, output_image):
    lines = cv2.HoughLines(input_image.copy(), 1, np.pi / 180, 150)
    if not repr(lines) == "None":
    	for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(output_image, (x1,y1), (x2,y2), (0,0,255), 2)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
mask = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

trackbars.setup()

while(True):
    ret, frame = cap.read()

    threshold1, threshold2, epsilon, sigma_color, sigma_space = trackbars.getUpdate()

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

    frame_composed = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)
    frame_composed[::] = 0

    color_means = []
    for bounding_rectangle in bounding_rectangles:
        mask[::] = 0
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(bounding_rectangle)]), 255)
        cv2.fillPoly(frame_composed, np.int32([cv2.boxPoints(bounding_rectangle)]), cv2.mean(frame, mask))

    if config.USE_HOUGH_TRAFO:
    	houghLineTransform(frame_edges, frame_composed)

    #for index, contour in enumerate(contours_filtered):
    #    cv2.drawContours(frame_composed, [contour], ALL_CONTOURS, color_means[index], 5)
    cv2.imshow('Canny Edge Threshold Differences', frame_composed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()