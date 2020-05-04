from numpy import *
import time
import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    def __init__(self, dataSet=None, k=None):
        self.dataSet = np.asarray(dataSet)
        self.k = k
        self.centers = None
        self.clustersIndexDist = None
        if dataSet is None or k is None:
            print("Dataset or clusters mustn't be NONE!")
            exit(1)

    # calculate Euclidean distance
    def euclDistance(self, vector1, vector2):
        return sqrt(sum(power(vector2 - vector1, 2)))  # distance

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
        while clusterChanged:
            clusterChanged = False
            # for each sample
            for i in range(numSamples):
                minDist = np.inf
                minIndex = 0
                # for each centroid; find the closest centroid
                for j in range(self.k):
                    distance = self.euclDistance(self.centers[j, :], self.dataSet[i, :])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                # update sample class
                if self.clustersIndexDist[i, 0] != minIndex:
                    clusterChanged = True
                    self.clustersIndexDist[i, :] = minIndex, minDist ** 2
            # update self.centers
            # print(self.centers)
            for j in range(self.k):
                # nonzero(self.clustersIndexDist[:, 0].A == j)[0] - take points for each clusters
                pointsInCluster = self.dataSet[nonzero(self.clustersIndexDist[:, 0].A == j)[0]]
                self.centers[j, :] = mean(pointsInCluster, axis=0)  # move centroid to mean values
            # print(self.centers)
    
        print('Congratulations, cluster complete!')
        return self.dataSet, self.k, self.centers, self.clustersIndexDist

    # show clusters only available with 2D data
    # self.centers - self.centers of clusters
    # self.clustersIndexDist: first column - number of cluster, second - distance
    def showClusters(self):
        numSamples, dim = self.dataSet.shape
        if dim != 2:
            print("The dimension of your data is not 2!")
            return 1

        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        if self.k > len(mark):
            print("k is too large! Make 'mark' more")
            return 1

        # draw all samples
        for i in range(numSamples):
            markIndex = int(self.clustersIndexDist[i, 0])
            plt.plot(self.dataSet[i, 0], self.dataSet[i, 1], mark[markIndex])

        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # draw the self.centers  
        for i in range(self.k):
            plt.plot(self.centers[i, 0], self.centers[i, 1], mark[i], markersize=10)
        plt.show()


points = [
     [1, 2],
     [2, 1],
     [3, 1],
     [5, 4],
     [5, 5],
     [6, 5],
     [10, 8],
     [7, 9],
     [11, 5],
     [14, 9],
     [14, 14],
     ]

kmeans = KMeans(np.asarray(points), 3)
arr = kmeans.kmeans()
print(arr)
kmeans.showClusters()
