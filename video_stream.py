ALL_CONTOURS = -1
OPTIMAL_DISTANCE = -1
COLOR_THRESHOLD = 30

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

def assignPixelToColor(pixel_color, colorset):
    for color in colorset:
        if np.allclose(pixel_color, color, atol=COLOR_THRESHOLD):
            return color
    return [0, 0, 0]

def SegmentImageByColor(image, colorset):
    for index in np.ndindex(image.shape[:2]):
        image[index] = assignPixelToColor(image[index], colorset)

def SegmentImageByColorHsv(image, colorset):
    image_black = np.zeros_like(image)
    for color in colorset:
        mask = cv2.inRange(image, tuple(channel - COLOR_THRESHOLD for channel in color), tuple(channel + COLOR_THRESHOLD for channel in color))
        kernel = np.ones((10, 10),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        locations = np.where(mask != 0)
        image_black[locations[0], locations[1]] = color
    return image_black


cap = cv2.VideoCapture(0)

colors = setupColorsFromUserInput(cap)

trackbars.setup()

while(True):
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

    frame = SegmentImageByColorHsv(frame, colors)

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

    cv2.imshow('Canny Edge Threshold Differences', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite("frame_edges.jpg", frame_edges)
        break


cap.release()
cv2.destroyAllWindows()