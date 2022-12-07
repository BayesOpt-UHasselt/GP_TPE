import ConfigSpace as CS
import sys
import matplotlib.pyplot as plt
from Problems.DTLZ import DTLZ1
from Problems.Problem import Problem
from MOTPE import MOTPE
import numpy as np
import pandas as pd
eps = sys.float_info.epsilon

def optimal_set():
    true_front = np.empty((0, 2))

    for f1 in np.linspace(0, 1, num=1000):
        f2 = 1 - np.sqrt(f1)
        true_front = np.vstack([true_front, [f1, f2]])

    true_front = pd.DataFrame(true_front, columns=['f1', 'f2'])  # convert to DataFrame
    return np.asarray(true_front)

seed                = 1990   #args.seed
num_variables       = 3     #args.num_variables
num_initial_samples = 10*num_variables-1    #args.num_initial_samples
num_max_evals       = 100   #args.num_max_evals
num_objectives      = 2     #args.num_objectives
k                   = 1     #args.k
num_candidates      = 24    #args.num_candidates
init_method         = 'lhs' #args.init_method
gamma               = 0.1   #args.gamma
base_configuration  = {
    'num_objectives': num_objectives,
    'num_variables': num_variables,
    'k': k,
    'seed': seed}
benchmark_name      = "DTLZ1" #args.benchmark_name

f       = DTLZ1(benchmark_name, base_configuration)
cs      = f.make_cs(CS.ConfigurationSpace(seed=seed))
problem = Problem(f, cs)
solver  = MOTPE(seed=seed)

history = solver.solve(
    problem,
    {'num_initial_samples': num_initial_samples,
     'num_max_evals': num_max_evals,
     'init_method': init_method,
     'num_candidates': num_candidates,
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