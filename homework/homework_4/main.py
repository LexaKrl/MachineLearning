import folium
from utils import load_geojson_points, calculate_distance, calculate_travel_time
from ant_colony import AntColony
import numpy as np

# Загрузка данных

points = load_geojson_points('data/export.geojson')


# Параметры

car_speed_kmh = 50
bus_speed_kmh = 35
pedestrian_speed_kmh = 5

speed_kmh = bus_speed_kmh

max_time = 1

# Создание матриц расстояний и времени
n_points = len(points)
distance_matrix = np.zeros((n_points, n_points))
time_matrix = np.zeros((n_points, n_points))
for i in range(n_points):
    for j in range(n_points):
        if i == j:
            continue
        distance = calculate_distance(points[i], points[j])
        time = calculate_travel_time(distance, speed_kmh)
        distance_matrix[i][j] = distance
        time_matrix[i][j] = time

# Запуск муравьиного алгоритма
ant_colony = AntColony(points, distance_matrix, time_matrix, max_time, speed_kmh)
best_path_indices, total_time, total_weight = ant_colony.run()

# Вывод результатов
print("Лучший маршрут:")
for idx in best_path_indices:
    print(f"{points[idx]['name']} (приоритет: {points[idx]['weight']})")
print(f"Общее время: {total_time/60:.2f} минут")
print(f"Суммарный приоритет: {total_weight}")

# Визуализация на карте
route_map = folium.Map(location=[points[0]['lat'], points[0]['lon']], zoom_start=13)
for idx in best_path_indices:
    folium.Marker(
        [points[idx]['lat'], points[idx]['lon']],
        popup=f"{points[idx]['name']} (приоритет: {points[idx]['weight']})"
    ).add_to(route_map)
route_coords = [[points[idx]['lat'], points[idx]['lon']] for idx in best_path_indices]
folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(route_map)
route_map.save("route_map.html")
