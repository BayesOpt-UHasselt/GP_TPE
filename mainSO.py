import ConfigSpace as CS
import sys
import matplotlib.pyplot as plt
from SO import SO
from Problems.Problem import Problem
from MOTPE import MOTPE

eps = sys.float_info.epsilon

seed                = 128   #args.seed
num_variables       = 1     #args.num_variables
num_initial_samples = 11*num_variables-1    #args.num_initial_samples
num_max_evals       = 150   #args.num_max_evals
num_objectives      = 1     #args.num_objectives

k                   = 1     #args.k
num_candidates      = 24    #args.num_candidates
init_method         = 'lhs' #args.init_method
gamma               = 0.25   #args.gamma
base_configuration  = {
    'num_objectives': num_objectives,
    'num_variables': num_variables,
    'k': k,
    'seed': seed}
benchmark_name      = "SO" #args.benchmark_name

f       = SO(benchmark_name, base_configuration)
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
if num_objectives == 1:
    fig = plt.figure(figsize=(8, 6))
    f1s = [fs['f1'] for fs in history['f']]
    xs = [fs['x'] for fs in history['x']]
    plt.scatter(xs, f1s)
    plt.title(benchmark_name)
    plt.xlabel('x')
    plt.ylabel('f')
    plt.grid()
    plt.show()
else:
#     print(history)
    print("Done!")