import ConfigSpace as CS
import sys
from sklearn.datasets import fetch_openml

from Hypervolume import HyperVolume
from Problems.SVMr import ML as ML_SVM
from Problems.DTr import ML as ML_DT
from Problems.MLr import ML as ML_MLP
from Problems.Problem import Problem
from MOTPE import MOTPE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

eps = sys.float_info.epsilon


variables_array = [5, 2, 5]
ML_names =  ["MLPr", "SVMr", "DTr"]
datasets_id = [504, 529, 42636, 550, 23516] #[189, 507, 504, 529, 42636, 550, 23516]

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
        gamma               = 0.3   #args.gamma

        # LOAD DATA
        dataset = fetch_openml(data_id=id)
        X = dataset.data
        y = dataset.target
        # X_train and y_train will be used in an internal cross validation process for the HPO
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        base_configuration  = {
            'num_objectives': num_objectives,
            'num_variables': num_variables,
            'replications': 10,
            'k': k,
            'seed': seed}

        if ML_algorithm == "MLPr":
            f = ML_MLP(ML_algorithm, X_train, y_train, X_test, y_test, base_configuration)
        elif ML_algorithm == "SVMr":
            f = ML_SVM(ML_algorithm, X_train, y_train, X_test, y_test, base_configuration)
        else: #DTr
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
        df.to_csv("../output/MOTPE_sklearn/meanObjectives_MOTPE_val_"+ML_algorithm+"_"+str(id)+"_g_"+str(gamma)+".csv")

        # Save the evaluation on test set for all the solutions found
        m = solver.test_evaluations()
        df = pd.DataFrame(m)
        df.to_csv("../output/MOTPE_sklearn/meanObjectives_MOTPE_test_"+ML_algorithm+"_"+str(id)+"_g_"+str(gamma)+".csv")

        rep = solver.history_replications()
        mmm = pd.DataFrame(rep)
        mmm.to_csv("../output/MOTPE_sklearn/replications_MOTPE_"+ML_algorithm+"_"+str(id)+"_g_"+str(gamma)+".csv")
print("Done")