import csv
import sys
import numpy as np
import random
from numpy import genfromtxt
from matplotlib import pyplot

def clusterData(points):
    return points

def main(argv):
    points = genfromtxt(argv[0], delimiter=",")
    clusterData(points)
    pyplot.scatter(points[:,0], points[:,1], c=points[:,2])
    pyplot.show()


if __name__ == "__main__":
    main(sys.argv[1:])