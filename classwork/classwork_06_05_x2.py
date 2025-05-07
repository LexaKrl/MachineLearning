from antsys import AntWorld
from antsys import AntSystem
import numpy as np
import random

def create_cities():
    print('cities:')
    print('| id |    x    |    y    |')
    cities = []
    for city in range(10):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        cities.append((city, x, y))
        print('|%4i|%9.4f|%9.4f|' % cities[city])

    return cities

def salesman_rules(start, end):
  return [((start[1]-end[1])**2+(start[2]-end[2])**2)**0.5]

def salesman_heuristic(path, candidate):
  return candidate.info

def salesman_cost(path):
  cost = 0
  for edge in path:
    cost+=edge.info
  return cost

def print_solution(sys_resp):
  print('total cost = %g' % sys_resp[0])
  print('path:')
  print('| id |    x    |    y    |--distance-->| id |    x    |    y    |')
  for edge in sys_resp[2]:
    print('|%4i|%9.4f|%9.4f|--%8.4f-->|%4i|%9.4f|%9.4f|' %
          (edge.start[0], edge.start[1], edge.start[2], edge.info, edge.end[0],
           edge.end[1], edge.end[2]))

def main():
    cities = create_cities()

    new_world = AntWorld(cities, salesman_rules, salesman_cost, salesman_heuristic)

    ant_opt = AntSystem(world=new_world, n_ants=50)

    ant_opt.optimize(50, 10)

    print_solution(ant_opt.g_best)



if __name__ == "__main__":
    main()