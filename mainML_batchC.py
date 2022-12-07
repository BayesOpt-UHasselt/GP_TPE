import ConfigSpace as CS
import sys
from sklearn.datasets import fetch_openml

from Hypervolume import HyperVolume
from Problems.SVM import ML as ML_SVM
from Problems.DT import ML as ML_DT
from Problems.ML import ML as ML_MLP
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

def compute_hypervolume(m, referencePoint):
    hv = HyperVolume(referencePoint)
    volume = hv.compute(m)
    return volume

variables_array = [5, 2, 5]
ML_names = ["MLP", "SVM", "DT"]
datasets_id = [1460, 980, 819, 871, 41146, 847, 803]

for id in datasets_id: # For each dataset
    for v_index in range(len(variables_array)):
        num_variables =variables_array[v_index] # Obtain the number of hyperparameters
        ML_algorithm = ML_names[v_index] # obtain the name of ML algorithm

        seed                = 1990   #args.seed
        num_initial_samples = 11*num_variables-1    #args.num_initial_samples
        num_max_evals       = 100   #args.num_max_evals
        num_objectives      = 2     #args.num_objectives
        k                   = 1     #args.k
        num_candidates      = 1000    #args.num_candidates
        init_method         = 'random' #args.init_method
        gamma               = 0.1   #args.gamma

        # LOAD DATA
        dataset = fetch_openml(data_id=id)
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

        if ML_algorithm == "MLP":
            f = ML_MLP(ML_algorithm, X_train, y_train, X_test, y_test, base_configuration)
        elif ML_algorithm == "SVM":
            f = ML_SVM(ML_algorithm, X_train, y_train, X_test, y_test, base_configuration)
        else:
            f = ML_DT(ML_algorithm, X_train, y_train, X_test, y_test, base_configuration)

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


        # Save all the solutions found
        m = solver.get_F()
        df = pd.DataFrame(m)
        df.to_csv("output/exploration_MOTPE_val_"+ML_algorithm+"_"+str(id)+".csv")

        # Save the evaluation on test set for all the solutions found
        m = solver.test_evaluations()
        df = pd.DataFrame(m)
        df.to_csv("output/exploration_MOTPE_test_"+ML_algorithm+"_"+str(id)+".csv")

        rep = solver.history_replications()
        mmm = pd.DataFrame(rep.reshape(rep.shape[0]*rep.shape[1],2))
        mmm.to_csv("output/replications_MOTPE_"+ML_algorithm+"_"+str(id)+".csv")
