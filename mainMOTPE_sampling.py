import ConfigSpace as CS
import sys
from Problems.ZDT import ZDT1
from Problems.Problem import Problem
from MOTPE import MOTPE
import numpy as np

eps = sys.float_info.epsilon

seed                = 128   #args.seed
num_initial_samples = 98    #args.num_initial_samples
num_max_evals       = 1000   #args.num_max_evals
num_objectives      = 2     #args.num_objectives
num_variables       = 9     #args.num_variables
replications        = 5
k                   = 1     #args.k
num_candidates      = 24    #args.num_candidates
init_method         = 'lhs' #args.init_method
gamma               = 0.1   #args.gamma
base_configuration  = {
    'num_objectives': num_objectives,
    'num_variables': num_variables,
    'replications': replications,
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
cs, i = solver.init_model(
    {'num_initial_samples': num_initial_samples,
     'num_max_evals': num_max_evals,
     'init_method': init_method,
     'num_candidates': num_candidates,
     'scalarize':True,
     'gamma': gamma},
        problem)
[sample, scores] = solver.sample(cs)
a = solver.history_replications()
solver.add_infill_point(problem, sample[0,:])
print("done")


