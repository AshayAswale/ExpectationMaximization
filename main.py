import csv
import sys
import numpy as np
import random
from numpy import genfromtxt
from matplotlib import pyplot
import time


def clusterData(points):
    return points


def plotPoints(points):
    pyplot.scatter(points[:, 0], points[:, 1], c=points[:, 2])
    pyplot.show()


def main(argv):
    points = genfromtxt(argv[0], delimiter=",")
    clusterData(points)
    kMeans(points, int(argv[1]))


def kMeans(points, n_clust):
    # # Delete when done
    # N = 10
    # points = np.random.rand(N, 2)

    points_dimension = len(points[0])
    centroids = np.zeros((n_clust, points_dimension))

    for i in range(n_clust):
        centroids[i] = random.choice(points)

    # Append a row of zeros.
    temp = np.zeros((len(points), points_dimension + 1))
    temp[:, :-1] = points
    points = temp

    # plotPoints(points)
    start_time = time.time()
    elapsed_time = time.time() - start_time
    while elapsed_time < 10:
        distances = getDistancesArray(points, centroids)
        assignClusters(points, distances)
        centroids = redefineCentroids(points, centroids)
        elapsed_time = time.time() - start_time
    plotPoints(points)


def redefineCentroids(points, centroids):
    centroids_reform = np.zeros((len(centroids), len(centroids[0])+1))
    for point in points:
        for i in range(len(centroids_reform[0])):
            centroids_reform[int(point[-1])][i] += point[i]
        centroids_reform[int(point[-1])][-1] += 1
    for point in centroids_reform:
        if point[-1] != 0:
            point[:-1] /= point[-1]
    return centroids_reform[:, :-1]


def assignClusters(points, distances):
    for i in range(len(points)):
        points[i, -1] = np.argmin(distances[i])


def getDistancesArray(points, centroids):
    distances = np.zeros((len(points), len(centroids)))
    for i in range(len(points)):
        for j in range(len(centroids)):
            distances[i, j] = getDistance(points[i], centroids[j])
    return distances


def getDistance(point, centroid):
    sum = 0
    for i in range(len(centroid)):
        sum += abs(point[i] - centroid[i])
    return sum


if __name__ == "__main__":
    main(sys.argv[1:])
