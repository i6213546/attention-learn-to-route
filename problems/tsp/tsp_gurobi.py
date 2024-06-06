#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import argparse
import numpy as np
from utils.data_utils import load_dataset, save_dataset
from gurobipy import *
import math, itertools, time

def solve_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

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

    # Dictionary of Euclidean distance between each pair of points
    # Here we can change into travelling cost
    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in dist.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)


    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)
    print(selected)
    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour

def solve_travelling_time_tsp(points, cost, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

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

    # Dictionary of Euclidean distance between each pair of points
    # Here we can change into travelling cost
    # dist = {(i,j) :
    #     math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
    #     for i in range(n) for j in range(i)}

    dist = {(i, j) : cost[i,j] for i in range(n) for j in range(n)}
    m = Model()
    m.Params.outputFlag = False

    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    # disable this feature as the cost now is no longer symmetric
    # for i,j in dist.keys():
    #     vars[j,i] = vars[i,j] # edge in opposite direction

    # Add degree-2 constraint (condition on binary variables: each node appears exact 2 times)
    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Optimize model
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)
    tour = subtour(selected)
    print('tour', tour)
    assert len(tour) == n

    return m.objVal, tour

def solve_all_gurobi(dataset, problem_type='euclidian', costset=None):
    results = []
    durations = []
    for i, instance in enumerate(dataset):
        print ("Solving instance {}".format(i))
        if problem_type == 'euclidian':
            start = time.time()
            result = solve_euclidian_tsp(instance)
            end   = time.time()
            durations.append(end-start)
        elif problem_type == 'travelling_time':
            cost = costset[i]
            start = time.time()
            solve_travelling_time_tsp(instance, cost)
            end   = time.time()
            durations.append(end-start)
        results.append(result)
    return results, durations

if __name__ == "__main__":
    problem        = 'tsp100'
    data_file_name = 'data/{}/test_location.pkl'.format(problem)
    data_file_cost = 'data/{}/test_cost.pkl'.format(problem)
    save_dir       = 'results/{}/gurobi'.format(problem)

    dataset = load_dataset(data_file_name)
    costset = load_dataset(data_file_cost)
    
    #[print(i) for i in range(len(dataset)) if len(dataset[i]) > 100]
    #results, durations = solve_all_gurobi(dataset=dataset[1:2], problem_type='travelling_time', costset=costset[1:2])
    results, durations = solve_all_gurobi(dataset=dataset[:1], problem_type='euclidian')
    print(results)
    print(durations)

    ############ run script ##################
    # python -m problems.tsp.tsp_gurobi