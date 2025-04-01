import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def main():
    data = []
    for i in range(3):
        np.random.seed(158)
        centerX = np.random.random()*5
        centerY = np.random.random()*5
        for j in range(30):
            data.append([random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)])
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

    dbscan = DBSCAN(eps = 0.5, min_samples = 3)
    predict = dbscan.fit_predict(data)

    plt.scatter(data[:, 0], data[:, 1], c = predict)
    plt.show()





if __name__ == "__main__":
    main()