ALL_CONTOURS = -1
OPTIMAL_DISTANCE = -1

import cv2, copy
import numpy as np
import collections
import config, trackbars
from contour import GeneralContour, RubiksCube

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

            
cap = cv2.VideoCapture("./Finalvideo.mp4") #"D:\\Dropbox\\Uni\\Master\\1. Semester\\IVU\\Final2.mp4")
ret, frame = cap.read()

trackbars.setup()
solution = np.zeros_like(frame)
cube = RubiksCube(solution)

while(True):
    ret, frame = cap.read()
        
    threshold1, threshold2, epsilon, sigma_color, sigma_space = trackbars.getUpdate()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_filtered = cv2.bilateralFilter(frame_gray, OPTIMAL_DISTANCE, sigma_color, sigma_space)
    frame_edges = cv2.Canny(frame_filtered, threshold1, threshold2)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(frame_edges, kernel, iterations=2)

    frame_edges, contours, hierarchies = cv2.findContours(frame_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    general_contours = [GeneralContour(contour, hierarchy, frame, True, epsilon/1000) 
    					for contour, hierarchy in zip(contours, hierarchies[0])]

    frame_composed = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)
    
    filtered_conts = []
    for contour in general_contours:
        if contour.is_candidate:
            center = contour.centroid
            found = 0
            for conttmp in filtered_conts:
                if(cv2.pointPolygonTest(conttmp.contour, (center[0], center[1]), False) > 0):
                    found = 1
            if found == 0:
                filtered_conts.append(contour)
    
    filtered_conts = cube.asFrCo(frame_composed, filtered_conts)

    for contour in filtered_conts:
        contour.draw(frame_composed)

    cv2.imshow('Canny Edge Threshold Differences', frame_composed)

    solution[::] = 0
    solution = cube.showFaces(solution)
    cv2.imshow('Solution', solution)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite("frame_edges.jpg", frame_composed)
        break


cap.release()
cv2.destroyAllWindows()