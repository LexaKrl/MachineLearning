import random as rd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def auto_silhouette(data, max_k=10):
    best_k = 2
    best_score = -1
    scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k

class MyKMeans:
    n_clusters = None
    dataset = None

    def __init__(self, dataset: Optional[np.ndarray] = None, n_clusters: int = 2):
        if dataset is not None and not isinstance(dataset, np.ndarray):
            self.dataset = np.array(dataset)
        else:
            self.dataset = dataset

        self.n_clusters = n_clusters
        self.centroids = []
        self.labels = []

    """
    Метод для измерения дистанций между двумя точками
    """

    def measure_length(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2) ** 2))

    """
    Определяем координаты центроидов
    """

    def init_centroids(self) -> List[np.ndarray]:
        centroids = []

        first_idx = rd.randint(0, len(self.dataset) - 1)
        centroids.append(self.dataset[first_idx])

        for _ in range(1, self.n_clusters):
            distances = []
            for point in self.dataset:
                min_dist = min(self.measure_length(point, centroid) for centroid in centroids)
                distances.append(min_dist)

            total_distance = sum(distances)
            probabilities = [d / total_distance for d in distances]

            new_centroid_idx = np.random.choice(len(self.dataset), p=probabilities)
            centroids.append(self.dataset[new_centroid_idx])

        return centroids


    """
    Присоединение точек к центроидам
    """

    def assign_clusters(self) -> List[int]:
        labels = []
        for point in self.dataset:
            distances = [self.measure_length(point, centroid) for centroid in self.centroids]
            closest_centroid_idx = np.argmin(distances)
            labels.append(closest_centroid_idx)
        return labels

    """
    Обновление координат центроидов
    """

    def update_centroids(self) -> bool:
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = self.dataset[np.array(self.labels) == i]
            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[i])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))

        if np.allclose(self.centroids, new_centroids):
            return False

        self.centroids = new_centroids
        return True

    def visualize(self, param1: int = 0, param2: int = 1):
        plt.scatter(self.dataset[:, param1], self.dataset[:, param2], c=self.labels, cmap='viridis')
        plt.scatter(np.array(self.centroids)[:, param1],
                    np.array(self.centroids)[:, param2],
                    marker='X', s=200, c='red')
        plt.show()

    def predict(self, X: List[List[float]]) -> List[int]:
        X = np.array(X)
        labels = []
        for point in X:
            distances = [self.measure_length(point, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return labels

    """
    fit() метод для обучения нашего алгоритма использует 
    assign_clusters(),
    update_centroids(),
    init_centroids()
    """

    def fit(self, max_iter: int = 100) -> None:
        if self.dataset is None:
            raise ValueError("Dataset is not provided")

        self.centroids = self.init_centroids()

        for _ in range(max_iter):
            self.labels = self.assign_clusters()

            # Визуалищация для второго задания
            self.visualize()

            if not self.update_centroids():
                break

    def visualize_all(self):
        length = self.dataset.shape[1]

        for i in range(length):
            for j in range(length):
                if i == j:
                    continue

                plt.subplot(length, length, i * length + j + 1)
                plt.scatter(self.dataset[:, j], self.dataset[:, i],
                            c=self.labels, cmap='viridis', alpha=0.7)
                plt.scatter(np.array(self.centroids)[:, j],
                            np.array(self.centroids)[:, i],
                            marker='X', s=100, c='red', linewidths=1)
        plt.tight_layout()
        plt.show()



def main():
    irises = load_iris()

    data = irises.data

    """
    Определили оптимальное количество кластеров = 3
    видно на картинке AVG_DISTANCES 
    """

    n_clusters = auto_silhouette(data)

    kmeans = MyKMeans(dataset=data, n_clusters= n_clusters)
    kmeans.fit()

    kmeans.visualize_all()
    return

if __name__ == "__main__":
    main()