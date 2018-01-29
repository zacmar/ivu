ALL_CONTOURS = -1
OPTIMAL_DISTANCE = -1

import cv2
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
colors = [(189.1, 204.27272727272728, 211.07272727272726), (57.12019230769231, 208.9375, 231.25480769230768), (83.23291925465838, 119.82919254658385, 230.45652173913044), (84.10588235294118, 168.1764705882353, 93.18823529411765), (22.851282051282052, 26.384615384615383, 151.03589743589743), (179.43571428571428, 127.85714285714286, 16.485714285714284)]

cap = cv2.VideoCapture("D:\\Dropbox\\Uni\\Master\\1. Semester\\IVU\\Final2.mp4")
ret, frame = cap.read()
mask = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

trackbars.setup()
# describes a deque holding all the contours and respective centroids
# for the last 5 frames

solution = np.zeros((480,640))
cube = RubiksCube()
cube.createCube()

emptyColor = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0)]

for i in range(6):
    cube.safeColor(i,emptyColor)
    
cube.showFaces(solution)
k = 0

while(True):
    ret, frame = cap.read()
    k+=1
    if k%2 == 0:
        continue
        
    threshold1, threshold2, epsilon, sigma_color, sigma_space = trackbars.getUpdate()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_filtered = cv2.bilateralFilter(frame_gray, OPTIMAL_DISTANCE, sigma_color, sigma_space)
    frame_edges = cv2.Canny(frame_filtered, threshold1, threshold2)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(frame_edges, kernel, iterations=2)

    # computing the contours and centroids for current frame
    frame_edges, contours, hierarchies = cv2.findContours(frame_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #frame_edges[::] = 0
    general_contours = [GeneralContour(contour, hierarchy, frame, True) for contour, hierarchy in zip(contours, hierarchies[0])]

    frame_composed = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2RGB)
    # frame_composed[::] = 0
    
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
    
    temp_frame_composed = frame_composed     
    frame_composed, filtered_conts = cube.asFrCo(frame_composed, filtered_conts)
    
    if(len(frame_composed)== 0):
        frame_composed = temp_frame_composed
        
    for contour in filtered_conts:
        contour.draw(frame_composed)
    
    
    template = cv2.imread('frame_edges.jpg',0)
    w, h = frame_edges.shape[::-1]
     
    #res = cv2.matchTemplate(frame_edges, template, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.09
    frame_new = frame_composed.copy()
    frame_new[::] = 0
    #loc = np.where( res >= threshold)
    #for pt in zip(*loc[::-1]):
    #    cv2.rectangle(frame_new, pt, (pt[0] + w, pt[1] + h), (255,0,255), 2)
    

    #cv2.imshow('res.png',frame_new)

    cv2.imshow('Canny Edge Threshold Differences', frame_composed)
    cv2.imshow('Solution', solution)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('w'):
        
        cv2.imwrite("frame_edges.jpg", frame_composed)
        break


cap.release()
cv2.destroyAllWindows()