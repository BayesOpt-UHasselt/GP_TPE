import optproblems.wfg
import sys
import numpy as np
import ConfigSpace.hyperparameters as CSH
eps = sys.float_info.epsilon

class WFG:
    def __init__(self, name, base_configuration):
        self.name = name
        self.base_configuration = base_configuration
        self.rng = np.random.RandomState(self.base_configuration["seed"])
        self.replications = []
        self.training_evaluations_cv = []  # the evaluation of each fold
        self.training_evaluations = []  # the evaluation of the whole training data
        self.test_evaluations = []  # the evaluation of the test data

    def __call__(self, configuration):
        num_variables = self.base_configuration['num_variables']
        num_objectives = self.base_configuration['num_objectives']
        k = self.base_configuration['k']
        function = {
            "WFG4": optproblems.wfg.WFG4,
            "WFG5": optproblems.wfg.WFG5,
            "WFG6": optproblems.wfg.WFG6,
            "WFG7": optproblems.wfg.WFG7,
            "WFG8": optproblems.wfg.WFG8,
            "WFG9": optproblems.wfg.WFG9,
        }

        arg = tuple(configuration["x" + str(i)] for i in range(1, num_variables + 1))
        f = function[self.name](num_objectives, num_variables, k)
        fitness = np.array(f.objective_function(arg))

        if 'sigma' in self.base_configuration:
            # Add noise according to the heterogeneus noise
            # fitness += self.random_state.normal(0, self.base_configuration['sigma'], len(fitness))
            constants = self.base_configuration['sigma']
            replications = self.base_configuration["replications"]
            f = np.zeros((replications, constants.shape[0]))
            for m in range(len(fitness)):  # for each objective

                a = constants[m, 0]
                t = constants[m, 1]

                for r in range(replications):
                    noise = self.rng.normal(loc=0, scale=(a * fitness[m] + a * t))  # (a*fr[m]+a*t)
                    f[r, m] = fitness[m] + noise

            fitness = np.mean(f, axis=0)
        return {f'f{i+1}': fitness[i] for i in range(len(fitness))}

    def create_hyperparameter(self, hp_type,
                              name,
                              lower=None,
                              upper=None,
                              default_value=None,
                              log=False,
                              q=None,
                              choices=None):
        if hp_type == 'int':
            return CSH.UniformIntegerHyperparameter(
                name=name, lower=lower, upper=upper, default_value=default_value, log=log, q=q)
        elif hp_type == 'float':
            return CSH.UniformFloatHyperparameter(
                name=name, lower=lower, upper=upper, default_value=default_value, log=log, q=q)
        elif hp_type == 'cat':
            return CSH.CategoricalHyperparameter(
                name=name, default_value=default_value, choices=choices)
        else:
            raise ValueError('The hp_type must be chosen from [int, float, cat]')

    def make_cs(self, cs):
        for i in range(1, self.base_configuration['num_variables'] + 1):
            var_name = "x" + str(i)
            hp = self.create_hyperparameter(
                hp_type="float",
                name=var_name,
                lower=0.0,
                upper=2.0 * i,
                default_value=0.0,
                log=False)
            cs.add_hyperparameter(hp)
        return cs

    def set_noise(self, constants):
        self.base_configuration['sigma'] = constants

    def num_objectives(self):
        return self.base_configuration['num_objectives']

    def num_replications(self):
        return self.base_configuration['replications']

    def get_replications(self):
        return self.replications

    def get_training_evaluations_cv(self):
        return self.training_evaluations_cv

    def get_training_evaluations(self):
        return self.training_evaluations

    def get_test_evaluations(self):
        return self.test_evaluations