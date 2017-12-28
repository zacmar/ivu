ALL_CONTOURS = -1
OPTIMAL_DISTANCE = -1

import cv2
import numpy as np
import collections
import config, trackbars
from contour import GeneralContour
from colorSetup import setupColorsFromUserInput

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

            cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


cap = cv2.VideoCapture(0)
ret, frame = cap.read()

frame_new = frame.copy()
colors = setupColorsFromUserInput()

for index, color in enumerate(colors):
    cv2.rectangle(frame_new, (100*index, 0), (100*index + 100, 400), color, -1)

cv2.imwrite('colors.jpg', frame_new)


trackbars.setup()


while(False):
    ret, frame = cap.read()

    threshold1, threshold2, epsilon, sigma_color, sigma_space = trackbars.getUpdate()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_filtered = cv2.bilateralFilter(frame_gray, OPTIMAL_DISTANCE, sigma_color, sigma_space)
    frame_edges = cv2.Canny(frame_filtered, threshold1, threshold2)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(frame_edges, kernel, iterations=2)

    # computing the contours and centroids for current frame
    frame_edges, contours, hierarchies = cv2.findContours(frame_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    general_contours = [GeneralContour(contour, hierarchy, frame) for contour, hierarchy in zip(contours, hierarchies[0])]

    frame_composed = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)
    # frame_composed[::] = 0

    for contour in general_contours:
        if contour.is_candidate:
            contour.draw(frame_composed)

    template = cv2.imread('frame_edges.jpg',0)
    w, h = frame_edges.shape[::-1]
     
    res = cv2.matchTemplate(frame_edges, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.09
    frame_new = frame_composed.copy()
    frame_new[::] = 0
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame_new, pt, (pt[0] + w, pt[1] + h), (255,0,255), 2)
    

    cv2.imshow('res.png',frame_new)

    cv2.imshow('Canny Edge Threshold Differences', frame_composed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite("frame_edges.jpg", frame_edges)
        break


cap.release()
cv2.destroyAllWindows()