import gurobipy as gp
from gurobipy import *
from utils.data_utils import load_dataset, save_dataset

problem        = 'tsp100'
data_file_name = 'data/{}/test_location.pkl'.format(problem)
data_file_cost = 'data/{}/test_cost.pkl'.format(problem)
save_dir       = 'results/{}/gurobi'.format(problem)

dataset = load_dataset(data_file_name)
costset = load_dataset(data_file_cost)
# Number of cities
n = len(dataset[0])

def solve_travelling_time_tsp(cost):

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle


    # Create a new model
    m = gp.Model("asymmetric_tsp")

    # Create decision variables
    vars = m.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Set the objective function
    m.setObjective(gp.quicksum(cost[i][j] * vars[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Add degree-2 constraint (condition on binary variables: each node appears exact 2 times)
    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Add subtour elimination constraints
    u = m.addVars(n, vtype=GRB.INTEGER, name="u")
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(u[i] - u[j] + n * vars[i, j] <= n - 1)

    # Optimize the model
    m.optimize()

    # Print the solution
    if m.status == GRB.OPTIMAL:
        solution = m.getAttr('x', vars)
        selected = tuplelist([(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5])
        tour = subtour(selected)
        print("Optimal tour:", tour)
        return m.objVal, tour
    else:
        print("No optimal solution found.")
        return None, None
