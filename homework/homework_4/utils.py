import json
import random
from geopy.distance import geodesic

def load_geojson_points(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = []
    for feature in data['features']:
        if feature['geometry']['type'] == 'Point':
            coords = feature['geometry']['coordinates']
            properties = feature.get('properties', {})
            name = properties.get('name', 'Unnamed')
            weight = random.randint(1, 10)  # случайный приоритет
            points.append({
                'name': name,
                'lat': coords[1],
                'lon': coords[0],
                'weight': weight
            })
    return points

def calculate_distance(point1, point2):
    coord1 = (point1['lat'], point1['lon'])
    coord2 = (point2['lat'], point2['lon'])
    return geodesic(coord1, coord2).meters

def calculate_travel_time(distance_meters, speed_kmh):
    speed_mps = speed_kmh * 1000 / 3600
    return distance_meters / speed_mps  # Время в секундах
