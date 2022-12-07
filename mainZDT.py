import ConfigSpace as CS
import sys
import matplotlib.pyplot as plt
from Problems.ZDT import ZDT1
from Problems.Problem import Problem
from MOTPE import MOTPE
import numpy as np


eps = sys.float_info.epsilon

seed                = 1990   #args.seed
num_variables       = 5     #args.num_variables
num_initial_samples = 11*num_variables-1    #args.num_initial_samples
num_max_evals       = 150   #args.num_max_evals
num_objectives      = 2     #args.num_objectives

k                   = 1     #args.k
num_candidates      = 1000    #args.num_candidates
init_method         = 'lhs' #args.init_method
gamma               = 0.1   #args.gamma
base_configuration  = {
    'num_objectives': num_objectives,
    'num_variables': num_variables,
    'replications': 5,
    'k': k,
    'seed': seed}
benchmark_name      = "ZDT1" #args.benchmark_name

f       = ZDT1(benchmark_name, base_configuration)
cs      = f.make_cs(CS.ConfigurationSpace(seed=seed))
problem = Problem(f, cs)
solver  = MOTPE(seed=seed)

range_objectives = np.asarray([[1, 10],
                              [0, 0]])
constants = solver.prepare_heterogeneus_noise(range_objectives)
history = solver.solve(
    problem,
    {'num_initial_samples': num_initial_samples,
     'num_max_evals': num_max_evals,
     'init_method': init_method,
     'num_candidates': num_candidates,
     'scalarize': True,
     'gamma': gamma})
if num_objectives == 2:
    fig = plt.figure(figsize=(8, 6))
    f1s = [fs['f1'] for fs in history['f']]
    f2s = [fs['f2'] for fs in history['f']]
    plt.scatter(f1s, f2s)
    plt.title(benchmark_name)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.grid()
    plt.show()
else:
#     print(history)
    print("Done!")