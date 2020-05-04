from numpy import *
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
from tqdm import tqdm
from random import shuffle


class KMeans:
    def __init__(self, dataSet=None, k=None, tol=0):
        self.dataSet = np.asarray(dataSet)
        self.k = k
        self.centers = None
        self.clustersIndexDist = None
        self.tol = tol
        if dataSet is None or k is None:
            print("Dataset or clusters mustn't be NONE!")
            exit(1)

    # calculate Euclidean distance
    def euclDistance(self, vector1, vector2):
        return sqrt(sum(power(vector2 - vector1, 2)))  # distance

    def calculate_dist(self, vec, minDist, minIndex):
        tmp = [self.euclDistance(self.centers[j, :], vec) for j in range(self.k)]
        if min(tmp) < minDist:
            minDist = min(tmp)
            minIndex = tmp.index(minDist)
        return minDist, minIndex

    # init self.centers with random samples
    def initCentroids(self, dataSet, k):
        numSamples, dims = dataSet.shape
        self.centers = zeros((k, dims))
        for i in range(k):
            index = int(random.uniform(0, numSamples))  # generate random numeric
            self.centers[i, :] = dataSet[index, :]  # keep point like center

    def kmeans(self):
        numSamples = len(self.dataSet)  # number of samples
        # first column stores which cluster this sample belongs to,  
        # second column stores the error between this sample and its centroid  
        self.clustersIndexDist = mat(zeros((numSamples, 2)))  # 2D-array to matrix
        # init self.centers
        self.initCentroids(self.dataSet, self.k)
        clusterChanged = True
        print("Alg started")
        # Find best center
        while clusterChanged:
            clusterChanged = False
            minDist = mat(full_like(self.clustersIndexDist[:, 0], np.inf))
            minIndex = mat(zeros_like(self.clustersIndexDist[:, 0]))
            # update minDist, minIndex
            for i in tqdm(range(len(self.clustersIndexDist))):
                minDist[i][0], minIndex[i][0] = self.calculate_dist(self.dataSet[i, :], minDist[i], minIndex[i])
            # if clusters changed
            if not array_equal(self.clustersIndexDist[:, 0], minIndex):
                clusterChanged = True
                self.clustersIndexDist[:, 0], self.clustersIndexDist[:, 1] = minIndex, np.power(minDist, 2)
            tmp = []
            for j in range(self.k):
                # nonzero(self.clustersIndexDist[:, 0].A == j)[0] - take points for each clusters
                pointsInCluster = self.dataSet[nonzero(self.clustersIndexDist[:, 0].A == j)[0]]
                mean_ = mean(pointsInCluster, axis=0)
                tmp += [self.euclDistance(self.centers[j, :], mean_)]
                self.centers[j, :] = mean_  # move centroid to mean values
            print(f"Max shift: {max(tmp)}")
            # early break
            if max(tmp) < self.tol:
                del tmp
                break
            del tmp
    
        print('Cluster complete!')
        return self.dataSet, self.k, self.centers, self.clustersIndexDist

    # show clusters only available with 2D data
    # self.centers - self.centers of clusters
    # self.clustersIndexDist: first column - number of cluster, second - distance
    def showClusters(self):
        numSamples, dim = self.dataSet.shape
        if dim != 2:
            print("The dimension of your data is not 2!")
            return 1
        markers = {'D': 'diamond', 's': 'square', '|': 'vline',
                   'x': 'x', '_': 'hline', 'd': 'thin_diamond',
                   'h': 'hexagon1', '+': 'plus', '*': 'star',
                   ',': 'pixel', 'o': 'circle', '.': 'point',
                   '1': 'tri_down', 'p': 'pentagon',
                   '3': 'tri_left', '2': 'tri_up', '4': 'tri_right',
                   'H': 'hexagon2', 'v': 'triangle_down',
                   '8': 'octagon', '<': 'triangle_left', '>': 'triangle_right'}
        markers = np.asarray(list(markers.keys()))
        colors = list(mcolors.TABLEAU_COLORS.values()) +\
                 list(mcolors.CSS4_COLORS.values())

        if self.k > len(markers):
            print("k is too large! Make 'markers' more")
            return 1

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(self.dataSet, '.')
        plt.savefig('source.png', dpi=450)
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(self.k):
            tmp = np.where(self.clustersIndexDist[:, 0] == i)[0]
            tmp_ = self.dataSet[tmp]
            plt.plot(tmp_[:, 0], tmp_[:, 1])
        plt.savefig('draw.png', dpi=450)
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(self.k):
            markIndex = i
            tmp = np.where(self.clustersIndexDist[:, 0] == i)[0]
            tmp_ = self.dataSet[tmp]
            np.random.shuffle(markers)
            plt.plot(tmp_[:, 0], tmp_[:, 1],
                     markers[markIndex],
                     c=colors[i], markersize=6)
            del tmp, tmp_

        # draw centers
        for i in range(self.k):
            plt.plot(self.centers[i, 0], self.centers[i, 1],
                     '^', c=colors[i], markersize=15)
        plt.savefig('res.png', dpi=450)
        plt.show()
