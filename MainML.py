import ConfigSpace as CS
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from Hypervolume import HyperVolume
# from KNN import ML
# from SVM import ML
from Problems.DT import ML
from Problems.Problem import Problem
from MOTPE import MOTPE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

eps = sys.float_info.epsilon


def pareto_front(points, level=0, index=False):

    points = points.copy()
    ranks = np.zeros(len(points))
    r = 0
    c = len(points)
    while c > 0:
        extended = np.tile(points, (points.shape[0], 1, 1))
        dominance = np.sum(np.logical_and(
            np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
            np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)
        points[dominance == 0] = 1e9  # mark as used
        ranks[dominance == 0] = r
        r += 1
        c -= np.sum(dominance == 0)
    if index:
#         return ranks
        return [i for i in range(len(ranks)) if ranks[i] == level]
    else:
        ind = [i for i in range(len(ranks)) if ranks[i] == level]
        return points[ind]

def optimal_set():
    true_front = np.empty((0, 2))

    for f1 in np.linspace(0, 1, num=1000):
        f2 = 1 - np.sqrt(f1)
        true_front = np.vstack([true_front, [f1, f2]])

    true_front = pd.DataFrame(true_front, columns=['f1', 'f2'])  # convert to DataFrame
    return np.asarray(true_front)

def compute_hypervolume(m, referencePoint):
    hv = HyperVolume(referencePoint)
    volume = hv.compute(m)
    return volume

seed                = 1990   #args.seed

num_variables       = 5     #args.num_variables
num_initial_samples = 11*num_variables-1    #args.num_initial_samples
num_max_evals       = 5   #args.num_max_evals
num_objectives      = 2     #args.num_objectives
k                   = 1     #args.k
num_candidates      = 1000    #args.num_candidates
init_method         = 'random' #args.init_method
gamma               = 0.1   #args.gamma
benchmark_name      = "banana" #args.benchmark_name

ML_algorithm = "DT"
# LOAD DATA

dataset = fetch_openml(name=benchmark_name)
X = dataset.data
y = dataset.target
# X_train and y_train will be used in an internal cross validation process for the HPO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)

base_configuration  = {
    'num_objectives': num_objectives,
    'num_variables': num_variables,
    'replications': 10,
    'k': k,
    'seed': seed}

f       = ML(benchmark_name, X_train, y_train, X_test, y_test, base_configuration)
cs      = f.make_cs(CS.ConfigurationSpace(seed=seed))
problem = Problem(f, cs)
solver  = MOTPE(seed=seed)

history = solver.solve(
    problem,
    {'num_initial_samples': num_initial_samples,
     'num_max_evals': num_max_evals,
     'init_method': init_method,
     'num_candidates': num_candidates,
     'scalarize': False,
     'gamma': gamma})

x  = solver.get_X()

# # Save all the solutions found
# m = solver.get_F()
# df = pd.DataFrame(m)
# df.to_csv("output/exploration_MOTPE_val_"+ML_algorithm+".csv")
#
# # Save the evaluation on test set for all the solutions found
# m = solver.test_evaluations()
# df = pd.DataFrame(m)
# df.to_csv("output/exploration_MOTPE_test_"+ML_algorithm+".csv")
#
# rep = solver.history_replications()
# mmm = pd.DataFrame(rep.reshape(rep.shape[0]*rep.shape[1],2))
# mmm.to_csv("output/replications_MOTPE_"+ML_algorithm+".csv")
#
# #### Case 1: Measures on cross validation
# m = solver.get_F()
# pf = pareto_front(m, index=True)
# pf = m[pf]
# volume=compute_hypervolume(pf, [1, 0])
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(m[:, 0], m[:, 1], color='orange', marker='+', zorder=m.shape[0], label="Feasible solutions")
# ax.scatter(pf[:, 0], pf[:, 1], color='k', s=50, label="Pareto front")
# plt.title("HV "+str(volume))
# plt.legend(loc='upper right')
# plt.show()
#
#
# m = solver.get_F()
# hv_temp = []
# for index in range(num_initial_samples, len(m)):
#     pf = pareto_front(m[:index], index=True)
#     pf = m[pf]
#     volume=compute_hypervolume(pf, [1, 0])
#     hv_temp.append(volume)
#
# fig, ax = plt.subplots(figsize=(10,10))
# plt.plot(hv_temp)
# plt.xlabel("Iterations (Number of infill points)")
# plt.ylabel("Hypervolume")
# plt.show()
#
#
# #### Case 2: Measures on training cross validation
#
# #### Case 3: Measures on training
#
# #### Case 4: Measures on test set
# m = solver.test_evaluations()
# pf = pareto_front(m, index=True)
# pf = m[pf]
# volume=compute_hypervolume(pf, [1, 0])
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(m[:, 0], m[:, 1], color='orange', marker='+', zorder=m.shape[0], label="Feasible solutions")
# ax.scatter(pf[:, 0], pf[:, 1], color='k', s=50, label="Pareto front")
# plt.title("HV "+str(volume))
# plt.legend(loc='upper right')
# plt.show()
