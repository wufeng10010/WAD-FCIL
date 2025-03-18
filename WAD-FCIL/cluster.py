import numpy as np
from sklearn.cluster import DBSCAN
import math
import copy

def cluster(models, num_clients):
    a = []
    for client in range(num_clients):
        client_task_id = models[client].task_id_list
        a.append(client_task_id)

    task_id0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(len(a)):
        task_id = copy.deepcopy(task_id0)
        for j in a[i]:
            task_id[j] = 1
        if i == 0:
            result = copy.deepcopy(task_id)
            continue
        else:
            result = np.vstack((result, task_id))


    dbscan = DBSCAN(eps=1., min_samples=3)

    clusters = dbscan.fit_predict(result)
    indices = np.where(clusters == -1)

    print(clusters)
    for i, cluster_label in enumerate(clusters):
        print(f"vector {result[i]} from {cluster_label}")

    return indices[0]


