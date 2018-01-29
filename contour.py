import math
import numpy as np
import cv2
import time, copy

CONTOUR_CLOSED = True
CONTOUR_OPEN = False
CHILD = 2
NO_CHILDREN = -1
ALL_CONTOURS = -1
COLOR_THRESHOLD = 40

class GeneralContour:
    def __init__(self, contour, hierarchy, colored_frame, original, epsilon):
        self.epsilon = epsilon
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
        b, g, r, a = cv2.mean(self._colored_frame, mask)
        return (b, g, r)

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
        self.approximation = cv2.approxPolyDP(self.contour, self.epsilon * perimeter, CONTOUR_CLOSED)
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

        if abs(cv2.contourArea(self.contour, True)) > 5000:
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


    def __init__(self, solution):
        self.initialized = False
        self.facletColors = dict()
        self.colorset = [(44, 74, 144),
					     (20, 29, 14),
					     (62, 105, 14),
					     (89, 150, 148),
                         (115, 75, 26),
					     (40, 39, 116)]
                         
        self.createCube()
        draworder = [4,0,1,2,5,8,7,6,3]
        emptyColor = [(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255)]

        for i in range(6):
            self.safeColor(i,copy.copy(emptyColor))
    
        self.showFaces(solution)
        self.initialized = True
        
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
                if not self.initialized:
                    cv2.drawContours(image, [facelet], 0, (255, 255, 255), 3)
                else:
                    cv2.drawContours(image, [facelet], 0, self.facletColors[faceNr][i], 3)
                i+=1
        return image
    
    def assignPixelToColor(self, pixel_color):
        min = 30000000.
        min_index = 0
        for index, color in enumerate(self.colorset):
            if np.linalg.norm(np.subtract(color, pixel_color)) < min:
                min = np.linalg.norm(np.subtract(color, pixel_color))
                min_index = index
        return self.colorset[min_index]
        
    def calc_dist(self, cent, centroids, distances):
        dist = 0
        for tempCent in centroids:
            dist += np.linalg.norm(cent-tempCent)
        distances.append(dist)
        return distances
        
    def retEachDistances(self, cent, centroids):
        distances = [np.linalg.norm(cent-tempCent) for tempCent in centroids]
        return distances, sorted(distances)
        
    def whoIsMyNeighbour(self, meLocation, meNr, centroids, alreadyAssigned):
        if meNr in alreadyAssigned:
            return alreadyAssigned
        alreadyAssigned.append(meNr)
        distances, sortedDist = self.retEachDistances(meLocation, centroids)
        
        for el in range(1,9):
            index = distances.index(sortedDist[el])
            if not index in alreadyAssigned:
                break
        return self.whoIsMyNeighbour(centroids[index],index,centroids,alreadyAssigned)
        


    def asFrCo(self, frame, conts):
        if len(conts) == 0:
            return [], []
        angle1 = 180*conts[0].angles[0]/math.pi
        treshhold = angle1*0.15
        upth = angle1 + treshhold
        loth = angle1 - treshhold
        indices = []
        for index, cont in enumerate(conts):
            angle = 180*cont.angles[0]/math.pi
            if angle > loth and angle < upth:
            	color = (255, 0, 0)
            	indices.append(index)
            else:
                color = (0, 255, 0)
                
            cv2.putText(frame, str("{:3.2f}".format(angle)), (cont.approximation[1][0][0], cont.approximation[1][0][1]), cv2.FONT_HERSHEY_PLAIN, 1,color,1,cv2.LINE_AA)
            #cv2.putText(frame, str("{:3.2f}".format(180*cont.angles[3]/math.pi)), (cont.approximation[0][0][0], cont.approximation[0][0][1]), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),2,cv2.LINE_AA)

        if len(indices) == 9: #here we found 9 facelets which appearently belong to the same face
            for index in indices:
                conts[index].color = self.assignPixelToColor((conts[index].color))
                cv2.drawContours(frame, [conts[index].contour], ALL_CONTOURS, conts[index].color, -1)
            conts = [conts[ind] for ind in indices]

            centroids = np.array([contour.centroid for contour in conts])
            print(centroids)
            print("-"*60)
            print(centroids[0])
            quit()
            distances = []
            for cent1 in centroids:
                distances = self.calc_dist(cent1, centroids, distances)

            middle = distances.index(min(distances)) #here we search for the middle centroid -> the one with the least distance to all the others
            distances = []
            distances = self.calc_dist(centroids[middle], centroids, distances)
            meNr = distances.index(max(distances))
            meLocation = centroids[meNr]
            alreadyAssigned = [middle]
            list_ = self.whoIsMyNeighbour(meLocation, meNr, centroids, alreadyAssigned)

            #the face has been scanned already, if the color of the middle facelet exists already
            middle_facelet = conts[list_[0]].color
            for faceNr in range(6):
                if self.facletColors[faceNr][4] == middle_facelet:
                    print("Already assigned")
                    return frame, conts
            for faceNr in range(6):
                if self.facletColors[faceNr][4] == (255,255,255):
                    for ind in range(len(list_)):
                        print(list_)
                        ok = list_[ind]
                        ok = conts[list_[ind]].color
                        self.facletColors[faceNr][ind] = conts[list_[ind]].color
                    return frame, conts
                
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