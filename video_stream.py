ALL_CONTOURS = -1

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def assignFaceletToFace(facelet, cube):
    # no facelets in any of the faces, faces is empty
    currentVec1 = facelet[0] - facelet[1]
    currentVec2 = facelet[1] - facelet[2]
    for face in cube:
        compareVec1 = face[0][0] - face[0][1]
        compareVec2 = face[0][1] - face[0][2]
        if np.allclose(currentVec1, compareVec1, 2) and np.allclose(currentVec2, compareVec2, 2):
            face.append(facelet)
            break
    else:
        cube.append([facelet])
        

def inList(array, list_):
    for element in list_:
        if np.array_equal(element, array):
            return True
    return False

while(True):
    ret, frame = cap.read()

    threshold1 = 50
    threshold2 = 20
    gray = cv2.bilateralFilter(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),9, 3,3)
    im2 = cv2.Canny(gray,threshold1,threshold2)
    im2, contours, hierarchy = cv2.findContours(im2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    indizes= []
    for cnt in contours:
        epsilon = 0.06*cv2.arcLength(cnt,True)
        indizes.append(cv2.approxPolyDP(cnt,epsilon,True))
    cont = [contour for contour in indizes if cv2.isContourConvex(contour) 
                                           and cv2.contourArea(contour) > 400
                                           and contour.shape[0] == 4]
    boundingRects = [cv2.minAreaRect(contour) for contour in cont]
    im2[::] = 0
    means = []
    cube = []
    for contour in cont:
        assignFaceletToFace(contour, cube)
    for index, boundingRect in enumerate(boundingRects): 
        mask = im2.copy()
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(boundingRect)]), (255, 255, 255))
        means.append(cv2.mean(frame, mask))
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    for index, contour in enumerate(cont):
        for ind, face in enumerate(cube):
            if inList(contour, face):
                cv2.putText(im2, str(ind), (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_PLAIN, 4,(255,255,255),2,cv2.LINE_AA)
                print(contour[0][0])
        cv2.drawContours(im2, [contour], ALL_CONTOURS, means[index], 5)
    cv2.imshow('frame', im2)
    cv2.imshow('grey', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def isContourClockwise(contour):
    return True if cv2.contourArea(contour) > 0 else False

