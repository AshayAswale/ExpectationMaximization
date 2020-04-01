import csv
import sys
import numpy as np
import random
from copy import deepcopy
from numpy import genfromtxt
from scipy import linalg
# from scipy.spatial import distance
from matplotlib import pyplot
import math
import time


def plotPoints(points, means, init_centroids, num=1):
    pyplot.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=2)
    for i in range(len(means)):
        pyplot.scatter(means[i, 0], means[i, 1], c=i, marker="*", s=90)
        # pyplot.scatter(init_centroids[i, 0], init_centroids[i, 1], c=4)
    # pyplot.show()
    name = 'EM_animation_3/books_read_'
    save_name = name + str(num) + '.png'
    print(save_name)
    pyplot.savefig(save_name)
    pyplot.close()


def main(argv):
    points = genfromtxt(argv[0], delimiter=",")
    solve(points, int(argv[1]))

    # # # Delete when done
    # N = 100
    # points = np.random.rand(N, 1)
    # solve(points, 1)


def solve(points, n_clust):
    points_dimension = len(points[0])

    # Append a row of zeros.
    temp = np.zeros((len(points), points_dimension + 1))
    temp[:, :-1] = points
    points = temp

    # Centroid with n-dimention and variance
    # means[ith centroid][x-cord, y-cord, std devn]
    means = np.zeros((n_clust, points_dimension))
    for i in range(n_clust):
        means[i] = random.choice(points[:, :-1])

    # Should be random symmetric matrix
    covariances = np.zeros((n_clust, points_dimension, points_dimension))
    for i in range(n_clust):
        covariances[i] = np.identity(points_dimension)

    # means = np.array([[20, 70, 5], [15, -10, 5], [-30, 15, 5]])

    expectationMaximisation(points, means, covariances)


def expectationMaximisation(points, means, covariances):
    p_centroid = np.full((len(means)), 1 / (len(means)))
    init_centroids = deepcopy(means)
    start_time = time.time()
    elapsed_time = time.time() - start_time
    i = 0
    # while elapsed_time < 10:
    while i < 10:
        p_x_cl_arr = getGaussianProbArray(points, means, covariances)
        cl_i_array = getProbOfBelonging(p_x_cl_arr, p_centroid)
        assignClusters(cl_i_array, points)
        print(cl_i_array)
        updateCentroids(points, means, covariances)
        elapsed_time = time.time() - start_time
        i += 1
        print(i)
        plotPoints(points, means, init_centroids, i)
    print(i)
    print(means)


def assignClusters(cl_i_array, points):
    for i in range(len(cl_i_array)):
        points[i, -1] = np.argmax(cl_i_array[i])


def updateCentroids(points, means, covariances):
    sums = np.zeros(len(means))
    loc_means = deepcopy(means)
    means.fill(0)
    covariances.fill(0)
    for i in range(len(points)):
        cluster = int(points[i, -1])
        sums[cluster] += 1
        means[cluster] += points[i, :-1]
        # covariances[cluster] += np.dot(np.atleast_2d(points[i, :-1] - loc_means[i]).T, (points[i, :-1] -loc_means[i]))
        diff = np.atleast_2d(points[i, :-1] - loc_means[cluster])
        covariances[cluster] += np.dot(diff.T, diff)

    for i in range(len(means)):
        means[i] /= sums[i]
        covariances /= sums[i]

    i = 0


def getProbOfBelonging(p_x_cl_arr, p_centroid):
    for point in p_x_cl_arr:
        sum = 0
        for i in range(len(point)):
            sum += point[i] * p_centroid[i]
            point[i] *= p_centroid[i]
        point[:] /= sum
    return p_x_cl_arr


def getGaussianProbArray(points, means, covariances):
    probabilities = np.zeros((len(points), len(means)))
    for i in range(len(points)):
        for j in range(len(means)):
            probabilities[i, j] = getGaussianProb(
                points[i], means[j], covariances[j])
    return probabilities


def getGaussianProb(point, centroid, covariance):
    dimention = len(centroid)
    denom = (1 / (((2 * math.pi) ** (dimention / 2))
                  * (abs(np.linalg.det(covariance)) ** 0.5)))
    expon = -1/2*(np.dot(np.dot(np.atleast_2d(point[:-1]-centroid),
                                (np.linalg.inv(covariance))), ((np.atleast_2d(point[:-1]-centroid)).T)))

    return (expon + np.log(denom))


if __name__ == "__main__":
    main(sys.argv[1:])
