import csv
import sys
import numpy as np
import random
from copy import deepcopy
from numpy import genfromtxt
from scipy import linalg
# from scipy.spatial import distance
# from matplotlib import pyplot
import math
import time


def plotPoints(points, means):
    pyplot.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=2)
    for i in range(len(means)):
        pyplot.scatter(means[i, 0], means[i, 1], c=i, marker="*", s=90)
    # pyplot.show()
    name = 'EM_animation_3/books_read_'
    save_name = name + str(plotPoints.counter) + '.png'
    print(save_name)
    pyplot.savefig(save_name)
    pyplot.close()
    plotPoints.counter += 1


plotPoints.counter = 0


def main(argv):
    points = genfromtxt(argv[0], delimiter=",")

    # If the n_clust is 0, range of clusters to be checked
    clusts_list = [2, 4]

    # Appending a column of zeros for cluster assigning
    points_dimension = len(points[0])
    temp = np.zeros((len(points), points_dimension + 1))
    temp[:, :-1] = points
    points = temp

    n_clust = int(argv[1])
    if n_clust is not 0:
        runtime = 10
        bic, ll, mean, variance = solve(points, n_clust, runtime)
    else:
        # Time
        runtime = 10 / (clusts_list[-1] - clusts_list[0] + 1)
        best_bic = math.inf
        n_clust = 0
        best_points = deepcopy(points)
        for i in range(clusts_list[0], clusts_list[-1]+1):
            bic, ll_curr, mean_curr, variance_curr = solve(points, i, runtime)
            if bic < best_bic:
                best_bic = bic
                ll = ll_curr
                mean = mean_curr
                variance = variance_curr
                best_points = deepcopy(points)
                n_clust = i
    print("############################################")
    print("Number of clusters:", n_clust)
    print("\nMeans:\n", mean)
    print("\nCovariance:\n", variance)
    print("\nLog-likelihood:\n", ll)
    print("############################################")


def solve(points, n_clust, runtime):
    # Initializing arrays
    points_dimension = len(points[0])-1
    means = np.zeros((n_clust, points_dimension))
    covariances = np.zeros((n_clust, points_dimension, points_dimension))

    # Best values of runs
    best_clusters = deepcopy(points)
    best_means = deepcopy(means)
    best_ll = -math.inf

    # Time
    solve.start_time = time.time()
    elapsed_time = time.time() - solve.start_time

    # Likelihood stopping threshold
    solve.threshold = 10
    solve.counter = 0

    while elapsed_time < runtime:

        # Centroid with n-dimention and variance
        # means[ith centroid][x-cord, y-cord, std devn]
        means.fill(0)
        for i in range(n_clust):
            means[i] = random.choice(points[:, :-1])

        # Should be random symmetric matrix
        covariances.fill(0)
        for i in range(n_clust):
            covariances[i] = np.identity(points_dimension)

        # means = np.array([[20, 70, 5], [15, -10, 5], [-30, 15, 5]])
        new_ll = expectationMaximisation(
            points, means, covariances, runtime)
        elapsed_time = time.time() - solve.start_time
        if new_ll > best_ll:
            best_clusters = deepcopy(points)
            best_means = deepcopy(means)
            best_covariance = deepcopy(covariances)
            # print("Best Updated from ", best_ll, " to ", new_ll)
            best_ll = new_ll
        solve.counter += 1
    # plotPoints(best_clusters, best_means)
    return getBIC(len(points), len(means), best_ll), best_ll, best_means, best_covariance


def getBIC(N, K, best_ll):
    return K*math.log(N)-2*best_ll


def expectationMaximisation(points, means, covariances, runtime):
    p_centroid = np.full((len(means)), 1 / (len(means)))
    elapsed_time = time.time() - solve.start_time
    prev_ll = math.inf
    curr_ll = 0
    # print("############")
    while elapsed_time < runtime and abs(curr_ll - prev_ll) > solve.threshold:
        prev_ll = curr_ll
        p_x_cl_arr, curr_ll = getGaussianProbArray(
            points, means, covariances, p_centroid)
        # print(curr_ll)
        cl_i_array = getProbOfBelonging(p_x_cl_arr, p_centroid)
        assignClusters(cl_i_array, points)
        updateCentroids(points, means, covariances)
        elapsed_time = time.time() - solve.start_time
        # solve.counter += 1
        # plotPoints(points, means)
    # print(means)
    return curr_ll


def assignClusters(cl_i_array, points):
    for i in range(len(cl_i_array)):
        points[i, -1] = np.argmax(cl_i_array[i])


def getLikelihood(points, means, covariances, p_centroid):
    likelihood = 0
    for i in range(len(points)):
        temp = 0
        for j in range(len(means)):
            temp += (p_centroid[j]) * getGaussianProb(
                points[i], means[j], covariances[j])
        likelihood += np.log(temp)
    return likelihood


def updateCentroids(points, means, covariances):
    sums = np.zeros(len(means))
    loc_means = deepcopy(means)
    means.fill(0)
    # covariances.fill(0)
    for i in range(len(points)):
        cluster = int(points[i, -1])
        sums[cluster] += 1
        means[cluster] += points[i, :-1]
        # covariances[cluster] += np.dot(np.atleast_2d(
        #     points[i, :-1] - loc_means[i]).T, (points[i, :-1] - loc_means[i]))
        diff = np.atleast_2d(points[i, :-1] - loc_means[cluster])
        covariances[cluster] += np.dot(diff.T, diff)

    for i in range(len(means)):
        if sums[i] is not 0:
            means[i] /= sums[i]
            covariances[i] /= sums[i]
        else:
            means[i] = random.choice(points[:, :-1])
            covariances[i] = np.identity(len(points[0])-1)

    i = 0


def getProbOfBelonging(p_x_cl_arr, p_centroid):
    for point in p_x_cl_arr:
        sum = 0
        for i in range(len(point)):
            sum += point[i] * p_centroid[i]
            point[i] *= p_centroid[i]
        if sum == 0.0:
            point[:] = 1/len(p_centroid)
        else:
            point[:] /= sum
    return p_x_cl_arr


def getGaussianProbArray(points, means, covariances, p_centroid):
    likelihood = 0
    p_x_cl_arr = np.zeros((len(points), len(means)))
    for i in range(len(points)):
        temp = 0
        for j in range(len(means)):
            gaus = getGaussianProb(
                points[i], means[j], covariances[j])
            p_x_cl_arr[i, j] = gaus
            temp += (p_centroid[j]) * gaus
        likelihood += np.log(temp)
    return p_x_cl_arr, likelihood


def getGaussianProb(point, centroid, covariance):
    dimention = len(centroid)
    denom = (1 / (((2 * math.pi) ** (dimention / 2))
                  * (abs(np.linalg.det(covariance)) ** 0.5)))
    diff = np.atleast_2d(point[:-1]-centroid)
    expon = -1/2 * \
        (np.dot(np.dot(diff, (np.linalg.inv(covariance))), ((diff).T)))

    # return expon + np.log(denom)
    return denom*math.exp(expon)


if __name__ == "__main__":
    main(sys.argv[1:])
