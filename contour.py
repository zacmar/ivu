import math
import numpy as np
import cv2

CONTOUR_CLOSED = True
CONTOUR_OPEN = False
CHILD = 2
NO_CHILDREN = -1
ALL_CONTOURS = -1


class GeneralContour:
    def __init__(self, contour, hierarchy, colored_frame):
        self.contour = contour
        self.hierarchy = hierarchy

        # candidates, at this point have to have the following characteristics:
        # * consist of 4 points in the polygon approximation -----> covered by _is_square
        # * have pairwise similar angles ---------------------´
        # * has to be convex --------------------------------´
        # * do not have any child contours
        # * cover a reasonable size of the input image
        
        self.is_candidate = self._is_candidate()
        if (self.is_candidate):
            self._bounding_rectangle = cv2.minAreaRect(self.contour)
            self._colored_frame = colored_frame
            self.color = self._compute_mean_color()

    def _compute_mean_color(self):
        mask = cv2.cvtColor(self._colored_frame.copy(), cv2.COLOR_RGB2GRAY)
        mask[::] = 0
        cv2.fillPoly(mask, np.int32([cv2.boxPoints(self._bounding_rectangle)]), 255)
        return cv2.mean(self._colored_frame, mask)

    def _is_candidate(self):
        if self._has_children():
            return False

        if not self._size_ok():
            return False
        
        if not self._is_square():
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

        angles = angles_in_square(self.approximation)
        if not angles_pairwise_equal(angles):
            return False
        
        return True

    def _size_ok(self):
        if abs(cv2.contourArea(self.contour)) < 400:
            return False

        print(abs(cv2.contourArea(self.contour)))

        if abs(cv2.contourArea(self.contour)) > 20000:
                return False

        return True

    def _has_children(self):
        if self.hierarchy[CHILD] != NO_CHILDREN:
            return False

        return True

    def draw(self, image):
        cv2.drawContours(image, [self.approximation], ALL_CONTOURS, self.color, 5)


class Face:
    def __init__(self, facelets):
        self.facelets = facelets

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
    print(a)
    b = cv2.norm(A, C)
    print(b)
    c = cv2.norm(A, B)
    print(c)
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

def angles_pairwise_equal(angles, threshold = 0.3):
    if angles[0] > angles[2] + threshold:
        print(False)
        return False
    if angles[0] < angles[2] - threshold:
        return False
    if angles[1] > angles[3] + threshold:
        return False
    if angles[1] < angles[3] - threshold:
        return False
    print("True")
    return True