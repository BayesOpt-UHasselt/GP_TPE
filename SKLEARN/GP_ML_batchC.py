import sys

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from pyswarms.single.global_best import GlobalBestPSO
from scipy.stats import norm
from numpy.random import uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ConfigSpace as CS

from Problems.Problem import Problem
from MOTPE import MOTPE
# Problems
from Problems.SVM import ML as ML_SVM
from Problems.DT import ML as ML_DT
from Problems.ML import ML as ML_MLP

random_seed = 1990

def acquisition_function(candidates, model, model_det, Z_min):
    Z_min = Z_min[0, 0]
    SK_gau = model.predict(candidates, return_std=False)
    [SK_gau_det, dk_mse] = model_det.predict(candidates, return_std=True)
    SK_gau = SK_gau
    mse = dk_mse  # np.sqrt(np.abs(dk_mse))

    mei = (Z_min - SK_gau) * norm.cdf((Z_min - SK_gau) / mse, 0, 1) + mse * norm.pdf((Z_min - SK_gau) / mse, 0, 1)
    return -mei[:, 0]


eps = sys.float_info.epsilon

ideal_point = np.asarray([0, -1])

variables_array = [5, 2, 5]
problem_names = ["MLP", "SVM", "DT"]
datasets_id = [997, 841, 53, 814, 770, 778, 41945, 980, 871, 41146, 847, 803]

#Options_PSO
options_pso = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}


for id_d in datasets_id: # For each dataset
    for id_problem in range(len(variables_array)):
        seed = random_seed  # args.seed
        num_variables = variables_array[id_problem]  # args.num_variables
        num_initial_samples = 11 * num_variables - 1  # args.num_initial_samples
        num_max_evals = 100  # args.num_max_evals
        num_objectives = 2  # args.num_objectives
        replications = 10
        benchmark_name = problem_names[id_problem]  # args.benchmark_name

        k = 1  # args.k
        num_candidates = 1000  # args.num_candidates
        init_method = 'random'  # args.init_method
        gamma = 0.3  # args.gamma

        base_configuration = {
            'num_objectives': num_objectives,
            'num_variables': num_variables,
            'replications': replications,
            'k': k,
            'seed': seed}

        #Load data
        dataset = fetch_openml(data_id=id_d)
        X = dataset.data
        y = dataset.target
        # X_train and y_train will be used in an internal cross validation process for the HPO
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Options_PSO
        if benchmark_name == "MLP":
            bounds = ([10, 5, 1, 0.0000001, 0.0000001], [1001, 1001, 7, 1, 1])
            f = ML_MLP(benchmark_name, X_train, y_train, X_test, y_test, base_configuration)
        elif benchmark_name == "SVM":
            bounds = ([0.1, 1], [2, 5])
            f = ML_SVM(benchmark_name, X_train, y_train, X_test, y_test, base_configuration)
        else:
            bounds = ([0, 0, 1, 1, 1], [21, 0.99, 11, 3, 2])
            f = ML_DT(benchmark_name, X_train, y_train, X_test, y_test, base_configuration)


        cs = f.make_cs(CS.ConfigurationSpace(seed=seed))
        problem = Problem(f, cs)
        solver = MOTPE(seed=seed)

        cs, i = solver.init_model(
            {'num_initial_samples': num_initial_samples,
             'num_max_evals': num_max_evals,
             'init_method': init_method,
             'num_candidates': num_candidates,
             'scalarize': False,
             'gamma': gamma,
             'newSampling': 0.01},
            problem)
        # data = solver.history_replications()
        # X = solver.get_X()
        hv_history = []
        W = solver.get_weights_scalarization()

        try:
            # Optimization procedure
            for it in range(num_max_evals):
                print("Iteration: ", it, "/", num_max_evals)

                data = solver.history_replications()[:, :2]  # Because the last column has the iteration number
                X = solver.get_X()

                # Prepare optimization
                Y_mean = np.zeros(X.shape[0])
                Vhat = np.zeros(X.shape[0])

                #### Scalarization
                # Randomly select a point
                weight = W[np.random.randint(0, W.shape[0]), :]

                j = 0
                for i in range(0, int(data.shape[0] / replications)):
                    point = np.asarray(data[i*replications:i*replications + replications, :])

                    ip_replicated = np.tile(ideal_point, (replications, 1))

                    pcheby_term1 = np.max((point - ip_replicated) * np.tile(weight, (replications, 1)), axis=1)
                    pcheby_term2 = np.sum((point - ip_replicated) * np.tile(weight, (replications, 1)), axis=1)
                    pcheby = pcheby_term1 + 0.05 * pcheby_term2
                    Y_mean[j] = np.mean(pcheby)
                    Vhat[j] = np.var(pcheby)  # because var already considers the division by N
                    j+=1
                # Vhat = np.asarray([Vhat]).T
                # Normalize objectives
                scaler = MinMaxScaler()
                Y_mean = scaler.fit_transform(np.asarray([Y_mean]).T)

                #     print(X.shape, Y_mean.shape, B.T.shape, Vhat.shape)
                kernel = C(0.1, (0.01, 10.0)) * RBF(1, (1e-2, 1e2))
                gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=Vhat,
                                               random_state=seed, n_restarts_optimizer=10).fit(X, Y_mean)
                # Obtain parameters from stochastic gpr to train a deterministic gpr
                # Then, these parameters are not optimized
                kernel_det = C(gpr.kernel_.get_params()['k1__constant_value'],
                               'fixed') * RBF(gpr.kernel_.get_params()['k2__length_scale'], 'fixed')
                gpr_det = GaussianProcessRegressor(kernel=kernel_det, normalize_y=True,
                                                   random_state=seed, n_restarts_optimizer=10).fit(X, Y_mean)

                #     ######### SELECTION ########
                #     Obtain the best point and evaluate it with SK
                index_best = np.argmin(Y_mean)
                # print("index_best", index_best)
                SK_gau = gpr.predict(np.asarray([X[index_best, :]]), return_std=False)

                print("Searching infill point...")
                optimizer = GlobalBestPSO(n_particles=300, dimensions=num_variables, options=options_pso,
                                          bounds=bounds)
                [mei, pos] = optimizer.optimize(acquisition_function, 100, verbose=False, model=gpr,
                                                model_det=gpr_det, Z_min=SK_gau)
                print("    Done")

                # Simulate the new point
                solver.add_infill_point(problem, pos, i)  # Evaluate the point
            # Save all the solutions found
            m = solver.get_F()
            df = pd.DataFrame(m)
            df.to_csv(
                "../output/GP_sklearn/meanObjectives_MOTPE_val_" + benchmark_name + "_" + str(id_d) + ".csv")

            # Save the evaluation on test set for all the solutions found
            m = solver.test_evaluations()
            df = pd.DataFrame(m)
            df.to_csv(
                "../output/GP_sklearn/meanObjectives_MOTPE_test_" + benchmark_name + "_" + str(id_d) + ".csv")

            rep = solver.history_replications()
            mmm = pd.DataFrame(rep)
            mmm.to_csv("../output/GP_sklearn/replications_MOTPE_" + benchmark_name + "_" + str(id_d) + ".csv")

            rep = solver.training_evaluations()
            df = pd.DataFrame(rep)
            df.to_csv("../output/GP_sklearn/training_MOTPE_" + benchmark_name + "_" + str(id_d) + ".csv")
        except Exception as e:
            print(e)
            # Save all the solutions found
            m = solver.get_F()
            df = pd.DataFrame(m)
            df.to_csv(
                "../output/GP_sklearn/meanObjectives_MOTPE_val_" + benchmark_name + "_" + str(id_d) + ".csv")

            # Save the evaluation on test set for all the solutions found
            m = solver.test_evaluations()
            df = pd.DataFrame(m)
            df.to_csv(
                "../output/GP_sklearn/meanObjectives_MOTPE_test_" + benchmark_name + "_" + str(id_d) + ".csv")

            rep = solver.history_replications()
            mmm = pd.DataFrame(rep)
            mmm.to_csv("../output/GP_sklearn/replications_MOTPE_" + benchmark_name + "_" + str(id_d) + ".csv")

            rep = solver.training_evaluations()
            df = pd.DataFrame(rep)
            df.to_csv("../output/GP_sklearn/training_MOTPE_" + benchmark_name + "_" + str(id_d) + ".csv")
