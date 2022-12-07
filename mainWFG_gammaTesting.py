import ConfigSpace as CS
import sys

from Hypervolume import HyperVolume
from Problems.WFG import WFG
from Problems.Problem import Problem
import numpy as np
from MOTPE import MOTPE
import pandas as pd
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

gammas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for g in gammas:
    seed                = 1990   #args.seed
    num_variables       = 5     #args.num_variables
    num_initial_samples = 11*num_variables-1    #args.num_initial_samples
    num_max_evals       = 130   #args.num_max_evals
    num_objectives      = 2     #args.num_objectives
    k                   = 1     #args.k
    replications        = 50
    num_candidates      = 1000    #args.num_candidates
    init_method         = 'lhs' #args.init_method
    gamma               = g   #args.gamma
    base_configuration  = {
        'num_objectives': num_objectives,
        'num_variables': num_variables,
    'replications': replications,
        'k': k,
        'seed': seed}
    benchmark_name      = "WFG4" #args.benchmark_name

    f       = WFG(benchmark_name, base_configuration)
    cs      = f.make_cs(CS.ConfigurationSpace(seed=seed))
    problem = Problem(f, cs)
    solver  = MOTPE(seed=seed)


    range_objectives = np.asarray([[3, 5],
                                  [0, 0]])
    constants = solver.prepare_heterogeneus_noise(range_objectives, rep_budget=replications)
    problem.set_noise(constants)
    history = solver.solve(
        problem,
        {'num_initial_samples': num_initial_samples,
         'num_max_evals': num_max_evals,
         'init_method': init_method,
         'num_candidates': num_candidates,
         'scalarize':False,
         'gamma': gamma})

    # Save the replications
    rep = solver.history_replications()
    mmm = pd.DataFrame(rep.reshape(rep.shape[0]*rep.shape[1],2))
    mmm.to_csv("notebook/PF_data/replications_noisy_MOTPE_WFG4_"+str(g)+".csv")

    # # Save the exploration
    m = solver.get_F()
    df = pd.DataFrame(m)
    df.to_csv("notebook/PF_data/exploration_noisy_MOTPE_WFG4_"+str(g)+".csv")

    print("Done!")