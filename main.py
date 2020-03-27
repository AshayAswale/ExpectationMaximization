import csv
import sys
import numpy as np
import random
from numpy import genfromtxt
from scipy.spatial import distance
# from matplotlib import pyplot
import math
import time


def clusterData(points):
    return points


def plotPoints(points):
    pyplot.scatter(points[:, 0], points[:, 1], c=points[:, 2])
    pyplot.show()


def main(argv):
    points = genfromtxt(argv[0], delimiter=",")
    clusterData(points)
    # # Delete when done
    N = 4
    points = np.random.rand(N, 2)
    solve(points, 2)
    # solve(points, int(argv[1]))


def solve(points, n_clust):
    points_dimension = len(points[0])

    # Append a row of zeros.
    temp = np.zeros((len(points), points_dimension + 1))
    temp[:, :-1] = points
    points = temp

    # Centroid with n-dimention and variance
    centroids = np.zeros((n_clust, points_dimension + 1))
    for i in range(n_clust):
        centroids[i] = random.choice(points)

    # Standard Deviation,
    centroids[:, -1] = 0.2
    expectationMaximisation(points, centroids)


def expectationMaximisation(points, centroids):
    # HARDCODING FUCK
    p_centroid = [0.2, 0.4, 0.4]
    start_time = time.time()
    elapsed_time = time.time() - start_time
    while elapsed_time < 10:
        p_x_cl_arr = getGaussianProbArray(points, centroids)
        cl_i_array = getProbOfBelonging(p_x_cl_arr, p_centroid)
        updateMean(points, cl_i_array, centroids)
        updateSD(points, cl_i_array, centroids)
        i = 0


def updateMean(points, cl_i_array, centroids):
    for i in range(len(centroids)):
        sum = 0
        temp = np.zeros((len(centroids[0])-1))
        for j in range(len(points)):
            sum += cl_i_array[j, i]
            for k in range(len(points[0])-1):
                temp[k] += cl_i_array[j, i] * points[j, k]
        centroids[i, :-1] = temp/sum


def getProbOfBelonging(p_x_cl_arr, p_centroid):
    for point in p_x_cl_arr:
        sum = 0
        for i in range(len(point)):
            sum += point[i] * p_centroid[i]
            point[i] *= p_centroid[i]
        point[:] /= sum

    return p_x_cl_arr


def getGaussianProbArray(points, centroids):
    probabilities = np.zeros((len(points), len(centroids)))
    for i in range(len(points)):
        for j in range(len(centroids)):
            probabilities[i, j] = getGaussianProb(points[i], centroids[j])
    return probabilities


def getGaussianProb(point, centroid):
    denom = (1 / (((2 * math.pi) ** 0.5)*centroid[-1]))
    expon = -((getDistance(point[:-1], centroid[:-1])
               ** 2)) / (2 * (centroid[-1] ** 2))
    return math.exp(expon)*denom


def getDistance(point, centroid):
    return distance.euclidean(point, centroid)


if __name__ == "__main__":
    main(sys.argv[1:])
