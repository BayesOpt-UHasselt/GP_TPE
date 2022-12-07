import optproblems.wfg
import sys
import numpy as np
import ConfigSpace.hyperparameters as CSH
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

eps = sys.float_info.epsilon


class ML:
    def __init__(self, name, X_train, y_train, X_test, y_test, base_configuration):
        self.name = name
        self.base_configuration = base_configuration
        self.rng = np.random.RandomState(self.base_configuration["seed"])
        # load dataset
        # dataset = fetch_openml(name=self.name)
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.replications = []
        self.training_evaluations_cv = []  # the evaluation of each fold
        self.training_evaluations = []  # the evaluation of the whole training data
        self.test_evaluations = []  # the evaluation of the test data

    def __call__(self, configuration):
        num_variables = self.base_configuration['num_variables']
        num_objectives = self.base_configuration['num_objectives']
        k = self.base_configuration['k']

        # Obtain the HP
        m = int(configuration["x1"])  # max_iter
        n = int(configuration["x2"])  # neurons
        li = int(configuration["x3"])  # lr_init
        b1 = configuration["x4"]  # b1
        b2 = configuration["x5"]  # b2

        kf = KFold(n_splits=self.base_configuration["replications"], shuffle=True, random_state=1)
        clf = MLPRegressor(activation='relu', learning_rate_init=1 / (10 ^ li), beta_1=b1, beta_2=b2,
                            solver="adam", hidden_layer_sizes=n, max_iter=m, random_state=1)

        self.replications = np.zeros((self.base_configuration["replications"], 2))
        self.training_evaluations_cv = np.zeros((self.base_configuration["replications"], 2))
        index_r = 0
        for train_indices, test_indices in kf.split(self.X, self.y):
            # Train
            clf.fit(self.X.values[train_indices], self.y.values[train_indices])

            # Test
            y_pred = clf.predict(self.X.values[test_indices])
            y_true = self.y.values[test_indices]

            error, r2 = self.evaluate_fitness(y_pred, y_true)
            self.replications[index_r, 0] = error
            self.replications[index_r, 1] = r2

            # Testing the training data
            y_pred = clf.predict(self.X.values[train_indices])
            y_true = self.y.values[train_indices]

            error, r2 = self.evaluate_fitness(y_pred, y_true)
            self.training_evaluations_cv[index_r, 0] = error
            self.training_evaluations_cv[index_r, 1] = r2
            index_r += 1

        # Compute the objectives using the training data
        y_pred = clf.predict(self.X.values)
        y_true = self.y.values

        error, r2 = self.evaluate_fitness(y_pred, y_true)
        self.training_evaluations = np.asarray([error, r2])

        # Compute the objectives using the test data
        y_pred = clf.predict(self.X_test.values)
        y_true = self.y_test.values

        error, r2 = self.evaluate_fitness(y_pred, y_true)
        self.test_evaluations = np.asarray([error, r2])

        fitness = np.asarray(np.mean(self.replications, axis=0))
        return {f'f{i + 1}': fitness[i] for i in range(len(fitness))}

    def evaluate_fitness(self, y_pred, y_true):
        # All the objective has to be minimized
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, -r2

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

        max_iter = self.create_hyperparameter(
            hp_type="int",
            name="x1",  # max_iter
            lower=10,
            upper=1001,
            default_value=50,
            log=False,
            q=1)
        neurons = self.create_hyperparameter(
            hp_type="int",
            name="x2",  # neurons
            lower=5,
            upper=1001,
            default_value=20,
            log=False,
            q=1)
        lr_init = self.create_hyperparameter(
            hp_type="int",
            name="x3",  # lr_init
            lower=1,
            upper=7,
            default_value=4,
            log=False,
            q=1)
        b1 = self.create_hyperparameter(
            hp_type="float",
            name="x4",  # b1
            lower=0.0000001,
            upper=1.0,
            default_value=0.9,
            log=False)
        b2 = self.create_hyperparameter(
            hp_type="float",
            name="x5",  # b2
            lower=0.0000001,
            upper=1.0,
            default_value=0.9,
            log=False)

        cs.add_hyperparameter(neurons)
        cs.add_hyperparameter(max_iter)
        cs.add_hyperparameter(lr_init)
        cs.add_hyperparameter(b1)
        cs.add_hyperparameter(b2)
        return cs

    def num_objectives(self):
        return self.base_configuration['num_objectives']

    def get_replications(self):
        return self.replications

    def get_training_evaluations_cv(self):
        return self.training_evaluations_cv

    def get_training_evaluations(self):
        return self.training_evaluations

    def get_test_evaluations(self):
        return self.test_evaluations

    def num_replications(self):
        return self.base_configuration['replications']
