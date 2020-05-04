from kmeans import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('train.csv', delimiter=',')
print(data.columns)


kmeans = KMeans(data[['place_latitude', 'place_longitude']], 10)
arr = kmeans.kmeans()
# print(arr)
kmeans.showClusters()




