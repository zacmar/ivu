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
colors = [(44.193627450980394, 74.07271241830065, 144.99264705882354), (20.247441860465116, 29.593488372093024, 14.581395348837209), (62.07008760951189, 105.57363370880267, 14.728410513141426), (89.2509671179884, 150.35444874274663, 148.47678916827854), (40.14426523297491, 39.47670250896057, 116.09498207885305), (115.79299645390071, 75.78147163120568, 26.62322695035461)]
cap = cv2.VideoCapture(1) #"D:\\Dropbox\\Uni\\Master\\1. Semester\\IVU\\Final2.mp4")
ret, frame = cap.read()
mask = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

trackbars.setup()
# describes a deque holding all the contours and respective centroids
# for the last 5 frames

solution = np.zeros((480,640,3))
cube = RubiksCube()
cube.createCube()

emptyColor = [(255,0,0),(255,255,255),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0)]

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