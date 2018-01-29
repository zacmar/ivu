import cv2
import numpy as np
import time
solution = np.zeros((400,1200,3))
centroids = np.array([[226, 273], [273, 265], [319, 256], [216, 226], [263, 218], [205, 178], [252, 170], [313, 208], [303, 160]])

def calc_dist(cent, centroids, distances):
    dist = 0
    for tempCent in centroids:
        dist += np.linalg.norm(cent-tempCent)
    distances.append(dist)
    return distances
    
def retEachDistances(cent, centroids):
    distances = [np.linalg.norm(cent-tempCent) for tempCent in centroids]
    return distances, sorted(distances)
    
def whoIsMyNeighbour(meLocation, meNr, centroids, alreadyAssigned):
    if meNr in alreadyAssigned:
        return alreadyAssigned
    alreadyAssigned.append(meNr)
    distances, sortedDist = retEachDistances(meLocation, centroids)
    
    for el in [1,2]:
        index = distances.index(sortedDist[el])
        if not index in alreadyAssigned:
            break
    return whoIsMyNeighbour(centroids[index],index,centroids,alreadyAssigned)
    



for centNr in range(9):
    cent = centroids[centNr]
    cv2.putText(solution, str(centNr), (cent[0],cent[1]), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),1,cv2.LINE_AA)

distances = []
for cent1 in centroids:
    distances = calc_dist(cent1, centroids, distances)

middle = distances.index(min(distances))
distances = []
distances = calc_dist(centroids[middle], centroids, distances)
meNr = distances.index(max(distances))
meLocation = centroids[meNr]
alreadyAssigned = [middle]
list = whoIsMyNeighbour(meLocation, meNr, centroids, alreadyAssigned)



cv2.imshow('Solution', solution)
cv2.waitKey(0)
cv2.destroyAllWindows()