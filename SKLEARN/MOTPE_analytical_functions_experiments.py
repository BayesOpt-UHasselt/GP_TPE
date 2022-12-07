import sys

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from pyswarms.single.global_best import GlobalBestPSO
from scipy.stats import norm
from numpy.random import uniform
from sklearn.preprocessing import MinMaxScaler
import ConfigSpace as CS

from Problems.Problem import Problem
from MOTPE import MOTPE
# Problems
from Problems.DTLZ import DTLZ
from Problems.WFG import WFG
from Problems.ZDT import ZDT1

# random_seed = 1990 # This was the seed with the original experiment

eps = sys.float_info.epsilon

ideal_point = np.asarray([0, 0])
problem_names = ["ZDT1", "WFG4", "DTLZ7"]
seeds = [8, 23, 45, 98, 123, 124, 139, 147, 458,
         4554, 5548, 85669]  # [ 8, 23, 45,98, 123,124, 139, 147, 458, 1990, 4554,  5548, 8569]

for random_seed in seeds:
    for id_problem in range(len(problem_names)):
        seed = random_seed  # args.seed
        num_variables = 5  # args.num_variables
        num_initial_samples = 11 * num_variables - 1  # args.num_initial_samples
        num_max_evals = 100  # args.num_max_evals
        num_objectives = 2  # args.num_objectives
        replications = 50

        k = 1  # args.k
        num_candidates = 1000  # args.num_candidates
        init_method = 'lhs'  # args.init_method
        base_configuration = {
            'num_objectives': num_objectives,
            'num_variables': num_variables,
            'replications': replications,
            'k': k,
            'seed': seed}
        benchmark_name = problem_names[id_problem]  # args.benchmark_name

        if benchmark_name == "ZDT1":
            f = ZDT1(benchmark_name, base_configuration)
            range_objectives = np.asarray([[1, 10],
                                           [0, 0]])
            hv_point = [1, 10]
            gamma = 0.3  # args.gamma
        elif benchmark_name == "WFG4":
            f = WFG(benchmark_name, base_configuration)
            range_objectives = np.asarray([[3, 5],
                                           [0, 0]])
            hv_point = [3, 5]
            gamma = 0.1  # args.gamma
        else:  # DTLZ7
            f = DTLZ(benchmark_name, base_configuration)
            range_objectives = np.asarray([[1, 23],
                                           [0, 0]])
            hv_point = [1, 23]
            gamma = 0.3  # args.gamma

        cs = f.make_cs(CS.ConfigurationSpace(seed=seed))
        problem = Problem(f, cs)
        solver = MOTPE(seed=seed)

        constants = solver.prepare_heterogeneus_noise(range_objectives, rep_budget=replications)
        solver_output = solver.solve(
            problem,
            {'num_initial_samples': num_initial_samples,
             'num_max_evals': num_max_evals,
             'init_method': init_method,
             'num_candidates': num_candidates,
             'scalarize': False,
             'gamma': gamma})

        # Save the replications
        rep = solver.history_replications()
        mmm = pd.DataFrame(rep)  # rep.reshape(rep.shape[0]*rep.shape[1],2)
        mmm.to_csv(
            "../output/MOTPE_sklearn/replications_MOTPE_" + benchmark_name + "_g_" + str(gamma) + "_s_" + str(
                random_seed) + ".csv")
        # Save the exploration
        m = solver.get_F()
        df = pd.DataFrame(m)
        df.to_csv(
            "../output/MOTPE_sklearn/meanObjectives_MOTPE_" + benchmark_name + "_g_" + str(gamma) + "_s_" + str(
                random_seed) + ".csv")
print("Done")
