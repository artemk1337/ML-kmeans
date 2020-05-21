from kmeans import KMeans
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('train.csv', delimiter=',')
print(data.columns)

kmeans_arr = []
kmeans_elbow_dist = []
x = []
for k in range(2, 10):
    kmeans = KMeans(data[['place_latitude', 'place_longitude']][:10000], k)
    arr, elbow = kmeans.kmeans()
    print(elbow)
    kmeans_arr.append(kmeans)
    kmeans_elbow_dist.append(elbow)
    x += [k]
    # print(arr)
    kmeans.showClusters()


plt.plot(x, kmeans_elbow_dist)
plt.savefig('elbow_graph.png', dpi=450)
plt.show()
