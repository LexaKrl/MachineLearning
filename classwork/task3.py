from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from fcmeans import FCM
import librosa


import matplotlib.pyplot as plt

def main():
    irises = load_iris()
    data = irises.data
    target = irises.target

    #plt.scatter(data[:,0], data[:, 1], c = target)
    #plt.show()

    pca = PCA(n_components = 2)
    pca.fit(data)
    data_pca = pca.transform(data)

    """
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data_pca)
    predict = kmeans.predict(data_pca)
    """

    cmeans = FCM(n_clusters=3)
    cmeans.fit(data)
    predict = cmeans.predict(data)

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c = predict)
    plt.show()


if __name__ == "__main__":
    main()