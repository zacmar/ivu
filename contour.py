import math
import cv2

CONTOUR_CLOSED = True
CONTOUR_OPEN = False
CHILD = 2
NO_CHILDREN = -1


class GeneralContour:
    def __init__(self, contour, hierarchy):
        self.contour = contour
        self.hierarchy = hierarchy

        # candidates, at this point have to have the following characteristics:
        # * consist of 4 points in the polygon approximation -----> covered by _is_square
        # * have pairwise similar angles ---------------------´
        # * do not have any child contours
        # * cover a reasonable size of the input image
        self.is_candidate = self._is_candidate()

    def _is_candidate(self):
        if self._has_children():
            return False

        if not self._size_ok():
            return False
        
        if not self._is_square():
            return False

        # Center of contour calculation by moments
        moments = cv2.moments(self.contour)
        self.center_of_mass = [moments["m10"] // moments["m00"],
                               moments["m01"] // moments["m00"]]

        return True

    def _is_square(self):
        perimeter = cv2.arcLength(self.contour, CONTOUR_CLOSED)
        self.approximation = cv2.approxPolyDP(self.contour, 0.1 * perimeter, CONTOUR_CLOSED)
        if len(self.approximation) != 4:
            return False

        angles = angles_in_square(self.approximation)
        if not angles_pairwise_equal(angles):
            return False
        
        return True

    def _size_ok(self):
        if cv2.contourArea(self.contour) < 400:
            return False

        return True

    def _has_children(self):
        if self.hierarchy[CHILD] != NO_CHILDREN:
            return False

        return True

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
        log.warning("get_angle: A %s, B %s, C %s, a %.3f, b %.3f, c %.3f" % (A, B, C, a, b, c))
        raise e
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle = math.acos(cos_angle)
    return angle

def angles_pairwise_equal(angles, threshold = 0.1):
    if angles[0] > angles[2] + threshold:
        return False
    if angles[0] < angles[2] - threshold:
        return False
    if angles[1] > angles[3] + threshold:
        return False
    if angles[1] < angles[3] - threshold:
        return False
    return True