import numpy as np


class AntColony:
    def __init__(self, points, distance_matrix, time_matrix, max_time, speed_kmh, n_ants=10, n_best=5, n_iterations=100, decay=0.1, alpha=1, beta=2):
        self.points = points
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.pheromone = np.ones(self.distance_matrix.shape) / len(points)
        self.all_inds = range(len(points))
        self.max_time = max_time * 3600  # Перевод в секунды
        self.speed_kmh = speed_kmh
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time = None
        all_weight = 0
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best)
            shortest = max(all_paths, key=lambda x: x[2])
            if shortest[2] > all_weight:
                shortest_path = shortest[0]
                all_time = shortest[1]
                all_weight = shortest[2]
            self.pheromone *=(1 - self.decay)
        return shortest_path, all_time, all_weight

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        path.append(start)
        total_time = 0
        total_weight = self.points[start]['weight']
        current = start
        while True:
            move = self.pick_move(self.pheromone[current], self.distance_matrix[current], visited)
            if move is None:
                break
            travel_time = self.time_matrix[current][move]
            if total_time + travel_time > self.max_time:
                break
            total_time += travel_time
            visited.add(move)
            path.append(move)
            total_weight += self.points[move]['weight']
            current = move
        return path, total_time, total_weight

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append(path)
        return all_paths

    def spread_pheromone(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[2], reverse=True)
        for path, time, weight in sorted_paths[:n_best]:
            for move in zip(path[:-1], path[1:]):
                self.pheromone[move] += weight / self.distance_matrix[move]

    def pick_move(self, pheromone, distances, visited):
        pheromone = np.copy(pheromone)
        distances = np.copy(distances)

        pheromone[list(visited)] = 0
        distances[list(visited)] = np.inf  # чтобы их не учитывать

        with np.errstate(divide='ignore', invalid='ignore'):
            heuristic = 1.0 / distances
            heuristic[np.isinf(heuristic)] = 0  # 1/inf → 0
            row = pheromone ** self.alpha * (heuristic ** self.beta)

        if np.sum(row) == 0 or np.isnan(row).any():
            return None

        norm_row = row / np.sum(row)
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move
