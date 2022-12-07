import optproblems.wfg
import sys
import numpy as np
import ConfigSpace.hyperparameters as CSH
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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
        c = configuration["x1"]  # C
        kernel_index = int(configuration["x2"])
        kernel_codes = ["linear", "poly", "rbf", "sigmoid"]# kernel
        kernel_choice = kernel_codes[kernel_index-1]

        kf = StratifiedKFold(n_splits=self.base_configuration["replications"])
        clf = SVC(C=c, kernel=kernel_choice, max_iter=500, random_state=1)

        self.replications = np.zeros((self.base_configuration["replications"], 2))
        self.training_evaluations_cv = np.zeros((self.base_configuration["replications"], 2))
        index_r = 0
        for train_indices, test_indices in kf.split(self.X, self.y):
            # Train
            clf.fit(self.X.values[train_indices], self.y.values[train_indices])

            # Test
            y_pred = clf.predict(self.X.values[test_indices])
            y_true = self.y.values[test_indices]

            error, recall = self.evaluate_fitness(y_pred, y_true)
            self.replications[index_r, 0] = error
            self.replications[index_r, 1] = recall

            # Testing the training data
            y_pred = clf.predict(self.X.values[train_indices])
            y_true = self.y.values[train_indices]

            error, recall = self.evaluate_fitness(y_pred, y_true)
            self.training_evaluations_cv[index_r, 0] = error
            self.training_evaluations_cv[index_r, 1] = recall
            index_r += 1

        # Compute the objectives using the training data
        # Train with all the training instances
        clf.fit(self.X.values, self.y.values)
        y_pred = clf.predict(self.X.values)
        y_true = self.y.values

        error, recall = self.evaluate_fitness(y_pred, y_true)
        self.training_evaluations = np.asarray([error, recall])

        # Compute the objectives using the test data
        y_pred = clf.predict(self.X_test.values)
        y_true = self.y_test.values

        error, recall = self.evaluate_fitness(y_pred, y_true)
        self.test_evaluations = np.asarray([error, recall])

        fitness = np.asarray(np.mean(self.replications, axis=0))
        return {f'f{i + 1}': fitness[i] for i in range(len(fitness))}

    def evaluate_fitness(self, y_pred, y_true):
        # All the objective has to be minimized
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        error = (fp + fn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) != 0 else 0
        return error, -recall

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

        C = self.create_hyperparameter(
            hp_type="float",
            name="x1",  # C
            lower=0.1,
            upper=2,
            default_value=1,
            log=False)
        kernel = self.create_hyperparameter(
            hp_type="float",
            name="x2",  # kernel
            lower=1,  # ["linear", "poly", "rbf", "sigmoid"],
            default_value=1,
            upper=5, # To use the round directly in the algorithm. If I use int with q, then the algorithm always suggest 3
            log=False)

        cs.add_hyperparameter(C)
        cs.add_hyperparameter(kernel)

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