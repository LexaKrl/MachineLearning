import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree

def main():
    file_path = "../AmesHousing.csv"

    data = pd.read_csv(file_path)
    print(data)
    data = data[["Lot Area", "Lot Frontage", "SalePrice"]]

    X = data[["Lot Area", "Lot Frontage"]]
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    Dtree = DecisionTreeRegressor(max_depth=6)

    Dtree.fit(X_train, y_train)

    print(Dtree.score(X_test, y_test))

    tree = plot_tree(Dtree)
    plt.show()

if __name__ == "__main__":
    main()