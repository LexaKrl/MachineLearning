from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

### Algorithm kmeans

def main():
    irises = load_iris()
    # print(irises)
    data = irises.data
    target = irises.target

    # plt.scatter(data[:,0], data[:,1], c = target)
    # plt.show()

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)
    predict = kmeans.predict(data)

    plt.scatter(data[:,0], data[:,1], c = predict)
    plt.show()

if __name__ == '__main__':
    main()
