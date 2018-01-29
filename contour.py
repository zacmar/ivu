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
MIDDLE_FACELET = 4


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


class RubiksCube:
    def __init__(self, solution):
        self.initialized = False
        self.facletColors = dict()
        self.colorset = [(58.13327120223672, 51.39608574091333, 254.93196644920783),
                         (0.0, 1.7061923583662715, 2.2674571805006587),
                         (0.09493087557603687, 247.2709677419355, 255.0),
                         (168.10998877665546, 71.6969696969697, 0.0),
                         (8.152564102564103, 153.74871794871794, 255.0),
                         (16.9991341991342, 149.7116883116883, 0.0)]
                         
        self.createCube()
        self.draworder = [4,0,3,6,7,8,5,2,1]
        emptyColor = [(255,255,255),(255,255,255),(255,255,255),
                      (255,255,255),(255,255,255),(255,255,255),
                      (255,255,255),(255,255,255),(255,255,255)]

        for i in range(6):
            self.safeColor(i, copy.copy(emptyColor))
    
        self.showFaces(solution)
        self.initialized = True

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
            self.facelets = self.createFace() + np.array([i*100,150])
            self.faces[i] = self.facelets
        self.faces[4] = self.createFace() + np.array([300,50])    
        self.faces[5] = self.createFace() + np.array([300,250]) 
        
    def showFaces(self,image):
        for id1, face in self.facletColors.items():
            for id2, facelet in enumerate(self.faces[id1]):
                    cv2.drawContours(image, [facelet], 0, face[id2], 3)
        return image
    
    def assignPixelToColor(self, pixel_color):
        min = 30000000.
        min_index = 0
        for index, color in enumerate(self.colorset):
            if np.linalg.norm(np.subtract(color, pixel_color)) < min:
                min = np.linalg.norm(np.subtract(color, pixel_color))
                min_index = index
        return self.colorset[min_index]
        
    def calc_dist(self, cent, centroids):
        dist = 0
        for tempCent in centroids:
            dist += np.linalg.norm(cent-tempCent)
        return dist
        
    def retEachDistances(self, cent, centroids):
        distances = [np.linalg.norm(cent-tempCent) for tempCent in centroids]
        return distances, sorted(distances)
        
    def whoIsMyNeighbour(self, meLocation, meNr, centroids, alreadyAssigned):
        if meNr in alreadyAssigned:
            return alreadyAssigned
        alreadyAssigned.append(meNr)
        distances, sortedDist = self.retEachDistances(meLocation, centroids)
        
        if(len(alreadyAssigned) != 2):
            for el in range(1,9):
                index = distances.index(sortedDist[el])
                if not index in alreadyAssigned:
                    break
        else:
            opt1 = centroids[distances.index(sortedDist[1])]
            opt2 = centroids[distances.index(sortedDist[2])]
            dir1 = opt1 - meLocation
            dir2 = opt2 - meLocation
            if dir1[0] > dir2[0]:
                index = distances.index(sortedDist[1])
            else:
                index = distances.index(sortedDist[2])

        return self.whoIsMyNeighbour(centroids[index],index,centroids,alreadyAssigned)
        
    def asFrCo(self, frame, conts):
        if len(conts) == 0:
            return conts
        angle_ref = conts[0].angles[0]
        treshhold = angle_ref * 0.15
        upth = angle_ref + treshhold
        loth = angle_ref - treshhold
        conts = [contour for contour in conts
                if contour.angles[0] > loth and contour.angles[0] < upth]

        if len(conts) == 9: #here we found 9 facelets which appearently belong to the same face

            for contour in conts:
                contour.color = self.assignPixelToColor((contour.color))
                contour.draw(frame)

            centroids = np.array([contour.centroid for contour in conts])
            for index, centroid in enumerate(centroids):
                cv2.putText(frame, str(index), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_PLAIN, 5,(0,0,255),2,cv2.LINE_AA)

            distances = [self.calc_dist(centroid, centroids)
                         for centroid in centroids]
            middle = distances.index(min(distances)) #here we search for the middle centroid -> the one with the least distance to all the others
            distances = []
            distances = [self.calc_dist(centroids[middle], centroids)]
            meNr = distances.index(max(distances))

            meLocation = centroids[meNr]
            alreadyAssigned = [middle]
            list_ = self.whoIsMyNeighbour(meLocation, meNr, centroids, alreadyAssigned)

            #the face has been scanned already, if the color of the middle facelet exists already
            middle_facelet = conts[list_[0]].color
            for _, face in self.facletColors.items():
                if face[MIDDLE_FACELET] == middle_facelet:
                    break
                elif face[MIDDLE_FACELET] == (255,255,255):
                    for ind in range(len(list_)):
                        face[self.draworder[ind]] = conts[list_[ind]].color
                    break
        
        return conts


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
    b = cv2.norm(A, C)
    c = cv2.norm(A, B)
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
        return False
    if angles[0] < angles[2] - threshold:
        return False
    if angles[1] > angles[3] + threshold:
        return False
    if angles[1] < angles[3] - threshold:
        return False
    return True