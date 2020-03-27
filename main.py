import csv
import sys
import numpy as np
import random
from numpy import genfromtxt
from scipy.spatial import distance
from matplotlib import pyplot
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
    # N = 100
    # points = np.random.rand(N, 2)
    # solve(points, 3)
    solve(points, int(argv[1]))


def solve(points, n_clust):
    points_dimension = len(points[0])

    # Append a row of zeros.
    temp = np.zeros((len(points), points_dimension + 1))
    temp[:, :-1] = points
    points = temp

    # Centroid with n-dimention and variance
    # centroids[ith centroid][x-cord, y-cord, std devn]
    centroids = np.zeros((n_clust, points_dimension + 1))
    for i in range(n_clust):
        centroids[i] = random.choice(points)
    centroids[:, -1] = 2
    # centroids = np.array([[20, 70, 5], [15, -10, 5], [-30, 15, 5]])

    expectationMaximisation(points, centroids)


def expectationMaximisation(points, centroids):
    p_centroid = np.full((len(centroids)), 1/(len(centroids)))
    start_time = time.time()
    elapsed_time = time.time() - start_time
    i = 0
    while elapsed_time < 10:
        # while i < 100:
        p_x_cl_arr = getGaussianProbArray(points, centroids)
        cl_i_array = getProbOfBelonging(p_x_cl_arr, p_centroid)
        updateCentroids(points, cl_i_array, centroids)
        elapsed_time = time.time() - start_time
        i += 1
    assignClusters(cl_i_array, points)
    plotPoints(points)
    print(i)
    print(centroids)


def assignClusters(cl_i_array, points):
    for i in range(len(cl_i_array)):
        points[i, -1] = np.argmax(cl_i_array[i])


def updateCentroids(points, cl_i_array, centroids):
    for i in range(len(centroids)):
        sum = 0
        mean = np.zeros((len(centroids[0]) - 1))
        variance = 0
        for j in range(len(points)):
            sum += cl_i_array[j, i]
            for k in range(len(points[0])-1):
                mean[k] += cl_i_array[j, i] * points[j, k]
            variance += (cl_i_array[j, i] *
                         ((getDistance(points[j, :-1], centroids[i, :-1]))**2))
        centroids[i, :-1] = mean / sum
        centroids[i, -1] = (variance/sum)**0.5


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
