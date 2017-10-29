ALL_CONTOURS = -1
OPTIMAL_DISTANCE = -1

import cv2
import numpy as np
import collections
import config, trackbars
import contour as contourtools

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

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
mask = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

trackbars.setup()
# describes a deque holding all the contours and respective centroids
# for the last 5 frames
frame_buffer = collections.deque(maxlen = 1)

while(True):
    ret, frame = cap.read()

    threshold1, threshold2, epsilon, sigma_color, sigma_space = trackbars.getUpdate()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_filtered = cv2.bilateralFilter(frame_gray, OPTIMAL_DISTANCE, sigma_color, sigma_space)
    frame_edges = cv2.Canny(frame_filtered, threshold1, threshold2)

    # computing the contours and centroids for current frame
    frame_edges, contours, hierarchies = cv2.findContours(frame_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    general_contours = [contourtools.GeneralContour(contour, hierarchy) for contour, hierarchy in zip(contours, hierarchies[0])]
    contours_approx = [cv2.approxPolyDP(contour, epsilon / 1000 * cv2.arcLength(contour, True), True)
                            for contour in contours]
    contours_filtered = [contour_ for contour_ in contours_approx
                            if contourtools.approx_is_square(contour_)]
    moments = [cv2.moments(contour) for contour in contours_filtered]
    centroids = [np.array([moment['m10'] // moment['m00'],
                              moment['m01'] // moment['m00']])
                   for moment in moments]

    # check if any given contour already exists in the frame buffer
    new_contours = []
    new_centroids = []
    for contour, centroid in zip(contours_filtered, centroids):
        in_buffer = False
        for i in range(len(frame_buffer) - 1):
            indizes_to_remove = []
            for j in range(len(frame_buffer[i][0])):
                if np.allclose(centroid, frame_buffer[i][1][j], 0.1):
                    print(j)
                    indizes_to_remove.append(j)
                    in_buffer = True
            for index in indizes_to_remove:
                pass
                #del frame_buffer[i][0][index]
                #del frame_buffer[i][1][index]
        if in_buffer == False:
            new_contours.append(contour)
            new_centroids.append(centroid)

    frame_buffer.append([new_contours, new_centroids])

    contours_to_draw = []
    for cur_frame in frame_buffer:
        for cont in cur_frame[0]:
            contours_to_draw.append(cont)

    bounding_rectangles = [cv2.minAreaRect(contour) for contour in contours_to_draw]

    frame_composed = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)
    frame_composed[::] = 0
    color_means = []
    for bounding_rectangle in bounding_rectangles:
        mask[::] = 0
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(bounding_rectangle)]), 255)
        color_means.append(cv2.mean(frame, mask))
        # cv2.fillPoly(frame_composed, np.int32([cv2.boxPoints(bounding_rectangle)]), cv2.mean(frame, mask))

    if config.USE_HOUGH_TRAFO:
        houghLineTransform(frame_edges, frame_composed)

    for index, contour in enumerate(contours_to_draw):
        cv2.drawContours(frame_composed, [contour], ALL_CONTOURS, color_means[index], 5)
    cv2.imshow('Canny Edge Threshold Differences', frame_composed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()