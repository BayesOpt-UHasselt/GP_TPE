class Problem:
    def __init__(self,
                 objective,
                 configspace,
                 n_objectives=1,
                 n_variables=None,
                 hyperparameters=None):
        self.objective = objective
        self.n_objectives = n_objectives
        self.n_variables = len(configspace.get_hyperparameters())
        self.configspace = configspace

    def __call__(self, x):
        return self.objective(x), self.get_replications(),\
               self.get_training_evaluations_cv(), self.get_training_evaluations(),\
               self.get_test_evaluations()


    def num_objectives(self):
        return self.objective.num_objectives()

    def set_noise(self, constants):
        return self.objective.set_noise(constants)

    def num_replications(self):
        return self.objective.num_replications()

    def get_replications(self):
        return self.objective.get_replications()

    def get_training_evaluations_cv(self):
        return self.objective.get_training_evaluations_cv()

    def get_training_evaluations(self):
        return self.objective.get_training_evaluations()

    def get_test_evaluations(self):
        return self.objective.get_test_evaluations()