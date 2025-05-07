from pyeasyga import pyeasyga


def fitness_function(individual, data):
    max_weight = data['max_weight']
    items = data['items']

    total_weight = 0
    total_value = 0
    for selected, item in zip(individual, items):
        if selected:
            total_weight += item[0]
            total_value += item[1]

    if total_weight > max_weight:
        return 0

    return total_value


def main():
    items = [(5, 12), (4, 10), (7, 15), (3, 8)]
    max_weight = 15

    data = {
        'items': items,
        'max_weight': max_weight
    }

    ga = pyeasyga.GeneticAlgorithm(data)

    ga.fitness_function = fitness_function

    ga.run()


    best_solution = ga.best_individual()
    print("Лучшее решение:", best_solution)


if __name__ == "__main__":
    main()