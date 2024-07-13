import argparse
import numpy as np
from utils.data_utils import load_dataset, save_dataset
import math, itertools, time, pickle
import gurobipy as gp
from gurobipy import GRB
import numpy as np

n=100
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # Get the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i in range(n) for j in range(n) if vals[i, j] > 0.5)
        
        # Find the subtours
        tour = find_subtour(selected)
        
        if len(tour) < n:
            # Add a subtour elimination constraint
            model.cbLazy(gp.quicksum(model._vars[i, j] for i in tour for j in tour if i != j) <= len(tour) - 1)

def find_subtour(edges):
    # Initialize a list to keep track of the nodes
    visited = [False] * n
    cycles = []
    lengths = []
    selected = [[] for _ in range(n)]
    
    for x, y in edges:
        selected[x].append(y)
        
    while True:
        # Find the first node that has not been visited
        current = next((i for i, v in enumerate(visited) if not v), -1)
        if current == -1:
            break
        
        thiscycle = []
        while not visited[current]:
            visited[current] = True
            thiscycle.append(current)
            current = selected[current][0]
        
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        
    return cycles[np.argmin(lengths)]

def gurobi(cost_matrix):
    # Number of cities
    n = len(cost_matrix)  # Adjust this as needed
    np.fill_diagonal(cost_matrix, 1000000)

    # Create a new model
    m = gp.Model("asymmetric_tsp")

    # Create variables
    vars = m.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Set objective
    m.setObjective(gp.quicksum(cost_matrix[i, j] * vars[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Add constraints
    # Each city is visited exactly once
    for i in range(n):
        m.addConstr(gp.quicksum(vars[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")
        m.addConstr(gp.quicksum(vars[j, i] for j in range(n) if j != i) == 1, name=f"in_{i}")

    # Optimize model with lazy constraints
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)

    # Retrieve the solution
    if m.status == GRB.OPTIMAL:
        vals = m.getAttr('x', vars)
        selected = gp.tuplelist((i, j) for i in range(n) for j in range(n) if vals[i, j] > 0.5)
        tour = find_subtour(selected)
        print("Optimal tour:", tour)
    else:
        print("No optimal solution found.")

    return tour

def nearest_neighbor(cost_matrix):
    n = len(cost_matrix)
    unvisited = set(range(n))
    tour = [0]  # Start from the first city
    unvisited.remove(0)

    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda city: cost_matrix[last][city])
        tour.append(next_city)
        unvisited.remove(next_city)

    #tour.append(tour[0])  # Return to the starting city
    return tour

if __name__ == "__main__":
    problem        = 'tsp100'
    data_file_name = 'data/{}/test_location.pkl'.format(problem)
    data_file_cost = 'data/{}/val_cost.pkl'.format(problem)

    dataset = load_dataset(data_file_name)
    costset = load_dataset(data_file_cost)
    tours_gurobi = []
    tours_nn = []
    for i in range(len(costset)):
        tour_gu = gurobi(costset[i])
        tours_gurobi.append(tour_gu)

        tour_nn = nearest_neighbor(costset[i])
        tours_nn.append(tour_nn)
    
    with open('outputs/tsp_100/gurobi/val_solution_as.pkl', 'wb') as f:
        pickle.dump(tours_gurobi, f, pickle.HIGHEST_PROTOCOL)
    
    with open('outputs/tsp_100/nn/val_solution_as.pkl', 'wb') as f:
        pickle.dump(tours_nn, f, pickle.HIGHEST_PROTOCOL)
    
    ############ run script ##################
    # python -m problems.tsp.gurobi