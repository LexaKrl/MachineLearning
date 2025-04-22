from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

# def generate_blobs(n_samples, centers):
#     data = []
#     for i in range(centers):
#         x = random.random()
#         y = random.random()
#         for j in range(int(n_samples/centers)):
#             data.append([random.gauss(x), random.gauss(y), centers])
#     return np.array(data)


def main():
    symptoms = pd.read_csv("..\\symptom.csv", sep = ";")
    disease = pd.read_csv("..\\disease.csv", sep = ";")

    patient = [1, 3, 5]
    probabilities = [1.] * (disease.shape[0] - 1)

    for i in range(disease.shape[0] - 1):
        probabilities[i] *= disease.loc[i][1]
        for j in range(symptoms.shape[0]):
            if j in patient:
                probabilities[i] *= symptoms.loc[j][i+1]
    print(disease.loc[np.argmax(probabilities)][0])

    print(symptoms.shape)
    print(disease.shape)

    print(f"Строк в symptoms: {len(symptoms)}")
    print(f"Строк в disease: {len(disease)}")

    # X_train, X_test, y_train, y_test = train_test_split(symptoms, disease)
    #
    # knn = KNeighborsClassifier(n_neighbors=10)
    #
    # knn.fit(X_train, y_train)
    # print(knn.score(X_test, y_test))


    # data, y = make_blobs(n_samples=100, n_features=5, centers=4)
    # print(data)
    # print(y)
    #
    # pca = PCA(n_components=2)
    # pca.fit(data)
    # data_transformed = pca.transform(data)
    #
    # X_train, X_test, y_train, y_test = train_test_split(data, y)
    # knn = KNeighborsClassifier(n_neighbors=10)
    #
    # knn.fit(X_train, y_train)
    # print(knn.score(X_test, y_test))


if __name__ == "__main__":
    main()