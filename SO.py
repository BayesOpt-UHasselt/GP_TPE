import optproblems.zdt
import sys
import numpy as np
import ConfigSpace.hyperparameters as CSH
eps = sys.float_info.epsilon

class SO:
    def __init__(self, name, base_configuration):
        self.name = name
        self.base_configuration = base_configuration

    def __call__(self, configuration):
        num_variables = self.base_configuration['num_variables']
        num_objectives = self.base_configuration['num_objectives']
        k = self.base_configuration['k']

        # Obtain the HP
        x = configuration["x"]

        sum1 = 0
        for i in range(1, 6):
            sum1 += i * np.cos((i + 1) * x + i)

        sum2 = 0
        for i in range(1, 6):
            sum2 += i * np.cos(i)

        # return sum1 * sum2

        fitness = np.array([sum1 * sum2])

        if 'sigma' in self.base_configuration:
            fitness += self.random_state.normal(0, self.base_configuration['sigma'], len(fitness))
        return {f'f{i + 1}': fitness[i] for i in range(len(fitness))}

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
            var_name = "x"
            hp = self.create_hyperparameter(
                hp_type="float",
                name=var_name,
                lower=-2.0,
                upper=1.0,
                default_value=0.0,
                log=False)
            cs.add_hyperparameter(hp)
        return cs

    def num_objectives(self):
        return self.base_configuration['num_objectives']