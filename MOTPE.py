import numpy as np
import pyDOE2
from TPESampler import TPESampler, default_weights
from Functions.GammaFunction import GammaFunction
import pandas as pd
import sys

class MOTPE:
    def __init__(self, seed=None):
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self._history = []
        self._history_replications = []
        self.linear_const = []

    def sample(self, cs):
        # ******* To call this method you have to calle first init_model
        # cs, i = self.init_model(parameters, problem)

        candidates = np.zeros((self.num_candidates, self.n_variables))
        scores = np.zeros((self.num_candidates, self.n_variables))
        split_cache = {}
        x = {}
        hp_index = 0
        for hp in cs.get_hyperparameters():
            sampler = TPESampler(hp,
                                 self._history,
                                 self.random_state,
                                 n_ei_candidates=self.num_candidates,
                                 gamma_func=GammaFunction(self.gamma),
                                 weights_func=default_weights,
                                 split_cache=split_cache,
                                 scalarize=self.scalarize,
                                 W=self.W)
            [candidates[:, hp_index], scores[:, hp_index]] = sampler.sample_from_lower(self.newSampling)

            split_cache = sampler.split_cache
            hp_index += 1
        # record = {'Trial': i, 'x': x, 'f': r}
        # self._history.append(record)
        #             print(record)

        return candidates, scores

    def init_model(self, parameters, problem):
        cs = problem.configspace
        self.replications =  problem.num_replications()
        hyperparameters = cs.get_hyperparameters()
        self.n_variables = problem.n_variables
        self.n_objectives = problem.num_objectives()
        seed = self.seed
        init_method = parameters['init_method']
        self.num_candidates = parameters['num_candidates']
        self.gamma = parameters['gamma']
        self.newSampling= parameters['newSampling'] if 'newSampling' in parameters else 1
        self.scalarize = parameters['scalarize']
        n_init_samples = parameters['num_initial_samples']
        # Weights for scalarization
        n_weights = 2*parameters['num_max_evals'];
        self.W = np.empty([n_weights, 2])
        self.W[:, 0] = np.linspace(0, 1, n_weights)
        self.W[:, 1] = np.abs(self.W[:, 0] - np.transpose(np.ones((n_weights, 1))))
        i = self.initial_sampling(cs, hyperparameters, init_method, n_init_samples, self.n_variables, problem)
        return cs, i

    def solve(self, problem, parameters):

        cs, i = self.init_model(parameters, problem)
        # np.argsort()
        while len(self._history) < parameters['num_max_evals']+parameters['num_initial_samples']:
            print("Optimization progress: ", np.round(len(self._history) / (parameters['num_max_evals']+parameters['num_initial_samples']) * 100, 2), "%")
            split_cache = {}
            x = {}
            for hp in cs.get_hyperparameters():
                sampler = TPESampler(hp,
                                     self._history,
                                     self.random_state,
                                     n_ei_candidates=parameters['num_candidates'],
                                     gamma_func=GammaFunction(parameters['gamma']),
                                     weights_func=default_weights,
                                     split_cache=split_cache,
                                     scalarize=parameters['scalarize'],
                                     W=self.W)
                x[hp.name] = sampler.sample()
                split_cache = sampler.split_cache
            f, r, t_cv, t, test = problem(x)
            # Obtain the noisy objectives
            if len(r) == 0:  # Function optimization. Generate noise now
                r = self.generate_noisy_objectives(f, problem.num_replications())
                t_cv = t = test =[] #It's used only for HPO

            self.update_history(i, f, x, r, t_cv, t, test)
            #             print(record)
            i += 1
        return self.history()

    def prepare_heterogeneus_noise(self, objectives, level=1, case=1, rep_budget=5):
        # Objectives is a 2xM matrix with the min and max of each objective

        M = objectives.shape[1]
        self.linear_const = np.zeros((M, 2))
        noise_sd_range = np.zeros((M, 2))

        for m in range(M):
            obj_i = objectives[:, m]
            rank_obj_i = np.sort(obj_i)[::-1]
            rank_obj_i[0] = 1 if m == 1 else 10
            rank_obj_i[-1] = 0
            range_obj_i = np.abs(rank_obj_i[0] - rank_obj_i[-1])

            # Bounds for noise s.d.
            if level == 1:
                lower_obj_i = 0.01 / np.sqrt(rep_budget) * range_obj_i
                upper_obj_i = 0.5 / np.sqrt(rep_budget) * range_obj_i
            elif level == 2:
                lower_obj_i = 0.5 / np.sqrt(rep_budget) * range_obj_i
                upper_obj_i = 1.5 / np.sqrt(rep_budget) * range_obj_i
            else:
                lower_obj_i = 1 / np.sqrt(rep_budget) * range_obj_i
                upper_obj_i = 2 / np.sqrt(rep_budget) * range_obj_i

            if case == 1:  # Best case
                b_obj_i = (rank_obj_i[0] * lower_obj_i - rank_obj_i[-1] * upper_obj_i) / (
                        upper_obj_i - lower_obj_i)
                a_obj_i = lower_obj_i / (rank_obj_i[-1] + b_obj_i)
            else:  # Worst case
                b_obj_i = (rank_obj_i[-1] * lower_obj_i - rank_obj_i[0] * upper_obj_i) / (
                        upper_obj_i - lower_obj_i)
                a_obj_i = lower_obj_i / (rank_obj_i[0] + b_obj_i)

            min_noise = a_obj_i * rank_obj_i[-1] + a_obj_i * b_obj_i
            max_noise = a_obj_i * rank_obj_i[0] + a_obj_i * b_obj_i

            self.linear_const[m, 0] = a_obj_i
            self.linear_const[m, 1] = b_obj_i
            noise_sd_range[m, 0] = min_noise
            noise_sd_range[m, 1] = max_noise

        return self.linear_const

    def update_history(self, i, f, x, replications=[], training_cv=[], training=[], test=[]):
        record = {'Trial': i, 'x': x, 'f': f, 't_cv': training_cv, 't':training, 'test':test}
        self._history.append(record)

        record_replications = {'Trial': i, 'r': replications}
        self._history_replications.append(record_replications)

    def initial_sampling(self, cs, hyperparameters, init_method, n_init_samples, n_variables, problem):
        i = 0
        if init_method == 'lhs':
            xs = pyDOE2.lhs(n_variables, samples=n_init_samples, criterion='maximin',
                            random_state=self.random_state)
        print("Sampling initial points: ")
        for i in range(n_init_samples):
            print("    ", np.round(i / n_init_samples * 100, 2), "%")
            if init_method == 'random':
                x = cs.sample_configuration().get_dictionary()
            elif init_method == 'lhs':
                # note: do not use lhs for non-real-valued parameters
                x = {d[0].name: (d[0].upper - d[0].lower) * d[1] + d[0].lower \
                     for d in zip(hyperparameters, xs[i])}
            else:
                raise Exception('unknown init_method')
            r, p, t_cv, t, test = problem(x)
            # Obtain the noisy objectives
            if len(p)==0: #Function optimization. Generate noise now
                p = self.generate_noisy_objectives(r, problem.num_replications())
                t_cv = t = test = []  # It's used only for HPO

            self.update_history(i, r, x, p, t_cv, t, test)
            #             print(record)
            i += 1
        # todo: implement sampling conditional parameters
        return i

    def history(self):
        return pd.DataFrame.from_dict(self._history)

    def history_replications(self):
        output = np.append(np.asarray(self._history_replications[0]["r"]),
                           np.tile(1, (self.replications,1)), axis=1)
        for index in range(1, len(self._history_replications)):
            exp = np.append(np.asarray(self._history_replications[index]["r"]),
                           np.tile(index+1, (self.replications,1)), axis=1)
            output = np.append(output, exp , axis=0)
        return np.asarray(output)

    def training_evaluations_cv(self):
        output = [self._history[0]["t_cv"]]
        for index in range(1, len(self._history)):
            output.append(self._history[index]["t_cv"])
        return np.asarray(output)

    def test_evaluations(self):
        output = [self._history[0]["test"]]
        for index in range(1, len(self._history)):
            output.append(self._history[index]["test"])
        return np.asarray(output)

    def training_evaluations(self):
        output = [self._history[0]["t"]]
        for index in range(1, len(self._history)):
            output.append(self._history[index]["t"])
        return np.asarray(output)

    def get_X(self):
        x_array = []

        for dic in self._history:
            row = dic["x"]
            temp = []
            for index in range(1, len(self._history[0]["x"]) + 1):
                hp = row["x" + str(index)]
                temp.append(str(hp))

            if len(x_array) == 0:
                x_array = np.asarray([temp])
            else:
                x_array = np.append(x_array,
                                    np.asarray([temp]),
                                    axis=0)
        return np.asarray(x_array)

    def get_F(self):
        f_array = []

        for dic in self._history:
            row = dic["f"]
            temp = np.zeros(len(self._history[0]["f"]))
            for index in range(1, len(self._history[0]["f"]) + 1):
                hp = row["f" + str(index)]
                temp[index - 1] = hp

            if len(f_array) == 0:
                f_array = np.asarray([temp])
            else:
                f_array = np.append(f_array,
                                    np.asarray([temp]),
                                    axis=0)
        return np.asarray(f_array)

    def generate_noisy_objectives(self, objectives, num_replications):
        output = np.zeros((num_replications, len(objectives)))

        array_objectives = np.zeros(len(objectives))
        # to keep the same order of all the objectives
        for index in range(len(objectives)):
            array_objectives[index] = objectives["f" + str(index + 1)]

        for m in range(len(array_objectives)):  # for each objective
            a = self.linear_const[m, 0]
            t = self.linear_const[m, 1]

            for r in range(num_replications):
                noise = np.random.normal(loc=0, scale=(a * array_objectives[m] + a * t))
                output[r, m] = array_objectives[m] + noise
        return output

    def get_weights_scalarization(self):
        return self.W

    def add_infill_point(self, problem, infill_point, i):
        # Convert the point
        point = {'x' + str(index+1): infill_point[index] for index in range(len(infill_point))}
        r, p, t_cv, t, test = problem(point)
        if len(p) == 0:  # Function optimization. Generate noise now
            p = self.generate_noisy_objectives(r, problem.num_replications())
            t_cv = t = test = []  # It's used only for HPO

        self.update_history(i, r, point, p, t_cv, t, test)
