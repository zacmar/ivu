import math
import numpy as np
import cv2
import time

CONTOUR_CLOSED = True
CONTOUR_OPEN = False
CHILD = 2
NO_CHILDREN = -1
ALL_CONTOURS = -1

class GeneralContour:
    def __init__(self, contour, hierarchy, colored_frame, original):
        self.contour = contour
        self.hierarchy = hierarchy
        self.original = original 
        # candidates, at this point have to have the following characteristics:
        # * consist of 4 points in the polygon approximation -----> covered by _is_square
        # * have pairwise similar angles ---------------------´
        # * has to be convex --------------------------------´
        # * do not have any child contours
        # * cover a reasonable size of the input image
        
        self.is_candidate = self._is_candidate()
        if (self.is_candidate == True):
            self._bounding_rectangle = cv2.minAreaRect(self.approximation)
            self._colored_frame = colored_frame
            self.color = self._compute_mean_color()

    def _compute_mean_color(self):
        mask = cv2.cvtColor(self._colored_frame.copy(), cv2.COLOR_RGB2GRAY)
        mask[::] = 0
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(self._bounding_rectangle)]), 255)
        return cv2.mean(self._colored_frame, mask)

    def _is_candidate(self):
        #if self._has_children():
        #    return False
        
        if not self._is_square():
            return False
            
        if not self._size_ok():
            return False
        


        # Center of contour calculation by moments
        moments = cv2.moments(self.contour)
        self.centroid = [moments["m10"] // moments["m00"],
                         moments["m01"] // moments["m00"]]

        return True

    def _is_square(self):
        perimeter = cv2.arcLength(self.contour, CONTOUR_CLOSED)
        self.approximation = cv2.approxPolyDP(self.contour, 0.06 * perimeter, CONTOUR_CLOSED)
        if len(self.approximation) != 4:
            return False

        if not cv2.isContourConvex(self.approximation):
            return False


        self._sort_contours()
        self.angles = angles_in_square(self.approximation)
        #if not angles_pairwise_equal(self.angles):
        #    return False
        
        return True

    def _size_ok(self):
        if abs(cv2.contourArea(self.contour, True)) < 500:
            return False

        #print(abs(cv2.contourArea(self.contour)))

        if abs(cv2.contourArea(self.contour, True)) > 2000:
                return False
        return True

    def _has_children(self):
        if self.hierarchy[CHILD] != NO_CHILDREN:
            return False

        return True

    def draw(self, image):
        cv2.drawContours(image, [self.approximation], ALL_CONTOURS, self.color, 3)

    def _sort_contours(self):
        """This method sorts the contours in the following way: leftmost point first, 
        if the leftmost point is ambiguous, we use the upper of the two. Next one is 
        nearest point to the right of it, again upper if ambiguous. We continue to sort
        in a Z - shape, Meaning the points are sorted 
        (top left, top right, bootom left, bottom right)."""

        points = np.array([approx[0] for approx in self.approximation])
        dt = [('col1', points.dtype),('col2', points.dtype)]
        view = points.ravel().view(dt)
        view.sort(order=['col1','col2'])
        if points[0][0] == points[1][0]:
                points[1], points[2] = points[2], points[1]

        if points[0][1] == points[1][1] and points[0][0] == points[3][0] \
        or points[0][0] == points[1][0] and points[0][1] == points[3][1]:
                input("jkhk")


        self.approximation[0][0] = np.array(points[0])
        self.approximation[1][0] = np.array(points[1])
        self.approximation[2][0] = np.array(points[2])
        self.approximation[3][0] = np.array(points[3])
        self.area = cv2.contourArea(self.approximation)
        #print("{}\n{}\n{}".format("*"*60, self.approximation, "*"*60))

class Face:
    def __init__(self):
        pass
    def get_missing(self, colored_frame):
        pass

class RubiksCube:
    def __init__(self):
        self.facletColors = dict()
        pass
        
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
    def safeColor(self, faceNr, colors):
        self.facletColors[faceNr] = colors

    def createFace(self):
        facelets = []
        facelet = np.array([[5,5],[25,5], [25,25], [5,25]])
        offX = offY = 26

        for i in range(9):
            facelets.append(facelet + np.array([offX*(i%3),offY*int(i/3)]))
        facelets = np.array(facelets)
        return facelets
        
    def createCube(self):
        self.faces = dict()
        for i in range(4):
            self.facelets = self.createFace() + np.array([i*100,200])
            self.faces[i] = self.facelets
        self.faces[4] = self.createFace() + np.array([300,100])    
        self.faces[5] = self.createFace() + np.array([300,300]) 
        
    def showFaces(self,image):
        for faceNr in range(6):
            i = 0
            for facelet in self.faces[faceNr]:
                print(self.facletColors[faceNr][i])
                cv2.drawContours(image, [facelet], ALL_CONTOURS, self.facletColors[faceNr][i],1)#(255,0,0)
                i+=1
    
    def asFrCo(self, frame, conts):
        tempcont1 = []
        tempcont2 = []
        if len(conts) == 0:
            return [], []
        angle1 = 180*conts[0].angles[0]/math.pi
        treshhold = angle1*0.15
        upth = angle1 + treshhold
        loth = angle1 - treshhold
        for cont in conts:
            angle = 180*cont.angles[0]/math.pi
            
            if angle > loth and angle < upth: 
                color = (255,0,0)
            else:
                color = (0, 255, 0)
                
            cv2.putText(frame, str("{:3.2f}".format(angle)), (cont.approximation[1][0][0], cont.approximation[1][0][1]), cv2.FONT_HERSHEY_PLAIN, 1,color,1,cv2.LINE_AA)
            #cv2.putText(frame, str("{:3.2f}".format(180*cont.angles[3]/math.pi)), (cont.approximation[0][0][0], cont.approximation[0][0][1]), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),2,cv2.LINE_AA)
        all_centroids = []    
        for cont in conts:
            all_centroids.append(cont.centroid)
        print ("_"*70)
        print (all_centroids)
        print ("_"*70)
        return frame, conts

    def ContToFacelets(self, frame, conts):
        pass
        
def angles_in_square(square):
    angles = []
    angles.append(angle_3_points(square[3][0], square[0][0], square[1][0]))
    angles.append(angle_3_points(square[0][0], square[1][0], square[2][0]))
    angles.append(angle_3_points(square[1][0], square[2][0], square[3][0]))
    angles.append(angle_3_points(square[2][0], square[3][0], square[0][0]))
    return angles

def angle_3_points(A, B, C):
    """ returns angle at B for the triangle A, B, C """
    a = cv2.norm(C, B)
    #print(a)
    b = cv2.norm(A, C)
    #print(b)
    c = cv2.norm(A, B)
    #print(c)
    try:
        cos_angle = (math.pow(a, 2) + math.pow(b, 2) - math.pow(c, 2)) / (2 * a * b)
    except ZeroDivisionError as e:
        # here two of the above three points are exactly the same...
        # not sure how this is possible and how to deal with it
        cos_angle = 0
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle = math.acos(cos_angle)
    return angle

def angles_pairwise_equal(angles, threshold = 1):
    if angles[0] > angles[2] + threshold:
        #print(False)
        return False
    if angles[0] < angles[2] - threshold:
        return False
    if angles[1] > angles[3] + threshold:
        return False
    if angles[1] < angles[3] - threshold:
        return False
    #print("True")
    return True