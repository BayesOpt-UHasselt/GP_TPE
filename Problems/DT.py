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
from sklearn.tree import DecisionTreeClassifier

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
        max_depth = int(configuration["x1"])
        min_samples_split = configuration["x2"]
        min_samples_leaf = int(configuration["x3"])
        max_features_index = int(configuration["x4"])
        max_features_codes = ["auto", "sqrt", "log2"]
        max_features = max_features_codes[max_features_index - 1]
        # criterion_index = int(configuration["x5"])
        # criterion_codes = ["gini", "entropy"]
        # criterion = criterion_codes[criterion_index - 1]
        criterion = configuration["x5"]

        kf = StratifiedKFold(n_splits=self.base_configuration["replications"])
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth if max_depth != 0 else None,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=1)

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

        max_depth = self.create_hyperparameter(
            hp_type="int",
            name="x1",  # max_depth
            lower=0,
            upper=21,
            default_value=0, #This is going to be replaced by None
            log=False,
            q=1)
        min_samples_split = self.create_hyperparameter(
            hp_type="float",
            name="x2",  # min_samples_split
            lower=0,
            default_value=0.5,
            upper=0.99,
            log=False)
        min_samples_leaf = self.create_hyperparameter(
            hp_type="int",
            name="x3",  # min_samples_leaf
            lower=1,
            default_value=1,
            upper=11,
            log=False,
            q=1)
        max_features = self.create_hyperparameter(
            hp_type="int",
            name="x4",  # max_depth [“auto”, “sqrt”, “log2”]
            lower=1,
            upper=3,
            default_value=1,
            log=False,
            q=1)
        criterion = self.create_hyperparameter(
            hp_type="cat",
            name="x5",  # criterion ["gini", "entropy"]
            default_value="entropy",
            choices=["gini", "entropy"])

        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        cs.add_hyperparameter(max_features)
        cs.add_hyperparameter(criterion)

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
