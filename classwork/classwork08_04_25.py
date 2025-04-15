import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fcmeans.cli import predict
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    data = pd.read_csv("..\\bikes_rent.csv")
    # sns.heatmap(data.corr(), annot=True, cmap="coolwarm")

    data = data.drop(["temp", "windspeed(mph)", "season"], axis = 1)
    X, y = data.drop(["cnt"], axis = 1), data["cnt"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    lr.score(X_test, y_test)
    print(lr.score(X_test, y_test))

    predict = lr.predict(X_test)
    A = np.sum((y_test - predict)**2)
    B = np.sum((y_test-y_test.mean())**2)
    print(1-A/B)

    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)

    



if __name__ == '__main__':
    main()