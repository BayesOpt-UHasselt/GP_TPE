import sys

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from numpy.random import uniform
from sklearn.preprocessing import MinMaxScaler
import ConfigSpace as CS

from Problems.Problem import Problem
from MOTPE import MOTPE
# Problems
from Problems.DTLZ import DTLZ
from Utilities import pareto_front, compute_hypervolume
from Problems.WFG import WFG
from Problems.ZDT import ZDT1

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

ideal_point = np.asarray([0, 0])
problem_names = ["ZDT1", "WFG4", "DTLZ7"]
# Options_PSO
bounds = (np.zeros(5), np.ones(5))
options_pso = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# lambdas = np.linspace(0.01, 0.1, 10)
# lambdas = np.append(lambdas, 1)
lambdas = [1]
gammas = [0.1, 0.2, 0.3]

seeds = [8, 23, 45, 98, 123, 124, 139, 147, 458,
         4554, 5548, 85669]  # [ 8, 23, 45,98, 123,124, 139, 147, 458, 1990, 4554,  5548, 8569]

for random_seed in seeds:

    for id_problem in range(len(problem_names)):
        for l in lambdas:
            for g in gammas:
                seed = random_seed  # args.seed
                num_variables = 5  # args.num_variables
                num_initial_samples = 11 * num_variables - 1  # args.num_initial_samples
                num_max_evals = 100  # args.num_max_evals
                num_objectives = 2  # args.num_objectives
                replications = 50

                k = 1  # args.k
                num_candidates = 1000  # args.num_candidates
                init_method = 'lhs'  # args.init_method
                gamma = g  # args.gamma
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
                elif benchmark_name == "WFG4":
                    f = WFG(benchmark_name, base_configuration)
                    range_objectives = np.asarray([[3, 5],
                                                   [0, 0]])
                    hv_point = [3, 5]
                else:  # DTLZ7
                    f = DTLZ(benchmark_name, base_configuration)
                    range_objectives = np.asarray([[1, 23],
                                                   [0, 0]])
                    hv_point = [1, 23]

                cs = f.make_cs(CS.ConfigurationSpace(seed=seed))
                problem = Problem(f, cs)
                solver = MOTPE(seed=seed)

                constants = solver.prepare_heterogeneus_noise(range_objectives, rep_budget=replications)
                cs, i = solver.init_model(
                    {'num_initial_samples': num_initial_samples,
                     'num_max_evals': num_max_evals,
                     'init_method': init_method,
                     'num_candidates': num_candidates,
                     'scalarize': False,
                     'gamma': gamma,
                     'newSampling': l},
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
                            point = np.asarray(data[i * replications:i * replications + replications, :])

                            ip_replicated = np.tile(ideal_point, (replications, 1))

                            pcheby_term1 = np.max((point - ip_replicated) * np.tile(weight, (replications, 1)), axis=1)
                            pcheby_term2 = np.sum((point - ip_replicated) * np.tile(weight, (replications, 1)), axis=1)
                            pcheby = pcheby_term1 + 0.05 * pcheby_term2
                            Y_mean[j] = np.mean(pcheby)
                            Vhat[j] = np.var(pcheby)  # because var already considers the division by N
                            j += 1
                        # Vhat = np.asarray([Vhat]).T
                        # Normalize objectives
                        scaler = MinMaxScaler()
                        Y_mean = scaler.fit_transform(np.asarray([Y_mean]).T)

                        #     print(X.shape, Y_mean.shape, B.T.shape, Vhat.shape)
                        kernel = C(0.1, (0.01, 10.0)) * RBF(1, (1e-2, 1e2))
                        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=Vhat, random_state=seed,
                                                       n_restarts_optimizer=10).fit(X, Y_mean)
                        # Obtain parameters from stochastic gpr to train a deterministic gpr
                        # Then, these parameters are not optimized
                        kernel_det = C(gpr.kernel_.get_params()['k1__constant_value'],
                                       'fixed') * RBF(gpr.kernel_.get_params()['k2__length_scale'], 'fixed')
                        gpr_det = GaussianProcessRegressor(kernel=kernel_det, normalize_y=True, random_state=seed,
                                                           n_restarts_optimizer=10).fit(X, Y_mean)

                        #     ######### SELECTION ########
                        #     Obtain the best point and evaluate it with SK
                        index_best = np.argmin(Y_mean)
                        # print("index_best", index_best)
                        SK_gau = gpr.predict(np.asarray([X[index_best, :]]), return_std=False)

                        print("Searching infill point...")
                        [sample_tpe, scores_tpe] = solver.sample(cs)
                        # Save the scores of this sample
                        df = pd.DataFrame(scores_tpe)
                        df.to_csv(
                            "../output/parameter_exploration_GP_MOTPE/SCORES_" + benchmark_name + "_gamma_" + str(
                                g) + "_l_" + str(l) + "_IT_" + str(it) + "_s_" + str(
                                random_seed) + ".csv")

                        # Select the point sampled from l that maximize the MEI
                        meis = acquisition_function(sample_tpe, gpr, gpr_det, SK_gau)
                        # Save the MEIs
                        df = pd.DataFrame(meis)
                        df.to_csv(
                            "../output/parameter_exploration_GP_MOTPE/MEIS_" + benchmark_name + "_gamma_" + str(
                                g) + "_l_" + str(l) + "_IT_" + str(it) + "_s_" + str(
                                random_seed) + ".csv")

                        # Select only the positive aggregated scores to reduce the probability to select a point with negative score (not good) and large mei
                        aggregated_scores = np.sum(scores_tpe, axis=1)
                        first_positive_index = list(map(lambda i: i > 0, aggregated_scores)).index(True)
                        meis = meis[first_positive_index:]
                        sample_tpe = sample_tpe[first_positive_index:, :]

                        # Since I am using the new sampling, the last configurations have the maximum score given by MOTPE.
                        # Therefore, the final configuration will be selected from gamma*sample
                        meis = meis[int((1 - l) * meis.shape[0]):]  # Use only the last candidates defined by lambda
                        sample = sample_tpe[int((1 - l) * sample_tpe.shape[0]):,
                                 :]  # Use only the last candidate defined by lambda

                        best_mei = meis.argmin(axis=0)
                        pos = sample[best_mei]
                        print("    Done")

                        # Save the new point
                        X = np.append(X, np.asarray([pos]), axis=0)

                        # Simulate the new point
                        solver.add_infill_point(problem, pos, i)  # Evaluate the point
                        data = solver.history_replications()  # To retrieve the evaluation of the new point
                        # Compute the means to obtain the Pareto front
                        unique = np.unique(data[:, 2], return_counts=False)
                        m = np.asarray([np.mean(data[data[:, 2] == di, :2], axis=0) for di in unique])

                        pf = pareto_front(m, index=True)
                        pf = m[pf]
                        # Compute the hypervolume
                        volume = compute_hypervolume(pf, hv_point)
                        hv_history.append(volume)
                    #     plot_evolution(X, XK, skriging_model_2, truthk, it)

                    # Save the replications
                    rep = solver.history_replications()
                    mmm = pd.DataFrame(rep)  # rep.reshape(rep.shape[0]*rep.shape[1],2)
                    mmm.to_csv(
                        "../output/parameter_exploration_GP_MOTPE/replications_GP_" + benchmark_name + "_gamma_" + str(
                            g) + "_l_" + str(l) + "_s_" + str(
                            random_seed) + ".csv")
                    # Save the exploration
                    m = solver.get_F()
                    df = pd.DataFrame(m)
                    df.to_csv(
                        "../output/parameter_exploration_GP_MOTPE/meanObjectives_GP_" + benchmark_name + "_gamma_" + str(
                            g) + "_l_" + str(l) + "_s_" + str(
                            random_seed) + ".csv")
                except Exception as e:
                    print(e)
                    # Save the replications
                    rep = solver.history_replications()
                    mmm = pd.DataFrame(rep)  # rep.reshape(rep.shape[0]*rep.shape[1],2)
                    mmm.to_csv(
                        "../output/parameter_exploration_GP_MOTPE/replications_GP_" + benchmark_name + "_gamma_" + str(
                            g) + "_l_" + str(l) + "_s_" + str(
                            random_seed) + ".csv")
                    # Save the exploration
                    m = solver.get_F()
                    df = pd.DataFrame(m)
                    df.to_csv(
                        "../output/parameter_exploration_GP_MOTPE/replications_GP_" + benchmark_name + "_gamma_" + str(
                            g) + "_l_" + str(l) + "_s_" + str(
                            random_seed) + ".csv")
