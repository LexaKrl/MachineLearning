import pandas as pd
import matplotlib.pyplot as plt

def main():
    dataset = pd.read_csv("train.csv")

    print(dataset[(dataset.Sex == "male") & (dataset.Pclass == 3)]["Survived"].mean())
    plt.show()

    # print(dataset.iloc[0:2])

    #print(dataset[(dataset.Age > 18) & (dataset.Survived == 1)])

    # plt.scatter(range(0, dataset.shape[0]), dataset.Age)
    # plt.show()

    # df["Name"] = ["Jack", "John", "Jeniffer", "Jane"]
    # df["Age"] = [20, 19, 17, 15]


if __name__ == '__main__':
    main()
